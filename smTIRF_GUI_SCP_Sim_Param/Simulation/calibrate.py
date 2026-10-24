"""
Fit the simulator's photophysics to the real movie's photometry (gate T3).

Separated from validate_against_real.py on purpose: the harness JUDGES, this
SEARCHES. Keeping them apart is what stops the acceptance criteria from quietly
becoming the fitting objective.

Only the four photometry quantities of section 4d are fitted here, because they
are the ones with a tractable analytic handle:

  * bg-subtracted brightness is LINEAR in peak amplitude, so one measurement gives
    the exact rescaling -- no search needed, just a ratio.
  * its CV is a function of the duty cycle alone: CV^2 = k/duty - 1, where k
    collects the jitter width and the measurement noise floor. Rather than model k,
    duty_needed_from_measurement() reads it off the current measurement, which
    absorbs the noise term instead of ignoring it.

Calibrate against the MEDIAN emitter, not the average one: section 4d reduces each
spot's trace and then takes the median across spots. See median_duty().

One limitation, found the hard way: under the SLOW switching the real movie turns out
to need (on/off periods of order 100 frames), an emitter goes through only one or two
cycles per movie, so its realised duty scatters far from the stationary value and the
analytic duty step below stops converging. For that regime set tau_on/tau_off directly
from the measured CV and use this script in report-only mode to check the result --
which is how config/real_calibrated_valband.yaml was produced.

Track duration (T4) is NOT fitted here. It is detector-mediated (trap 7): it
depends on how blink off-times interact with SCP's 3-frame tracking gap and on
per-frame detection probability, so it has no closed form. This script reports it
from one detector run per iteration and prints which direction tau_bleach should
move; the choice stays a human one.

    python calibrate.py                      # report the current config's fit
    python calibrate.py --fit-amplitudes     # rescale amplitudes, write the config
    python calibrate.py --fit-amplitudes --fit-duty --iterations 3
    python calibrate.py --with-detector      # add the T4 diagnostic (slow)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUTF8", "1")

import numpy as np
import yaml

import measurements as ms
import photophysics as pp
import simulate as sm

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "real_calibrated.yaml"
DEFAULT_REFERENCE = HERE / "config" / "real_movie_reference.json"

CLASSES = (("target", "target"), ("off_target", "off_target"))


def blink_duty(params):
    """Stationary on-fraction of the two-state blink chain."""
    tau_on = params.get("tau_on") or 0.0
    tau_off = params.get("tau_off") or 1.0
    if tau_on <= 0:
        return 1.0
    return float((1.0 / tau_off) / (1.0 / tau_off + 1.0 / tau_on))


def median_duty(params, n_frames):
    """Duty of the MEDIAN emitter, analytically -- the quantity to calibrate.

    Section 4d reduces each spot's own trace and then takes the median across
    spots, so it is the median emitter that sets the reported CV, not the average
    one. For an exponential bleach time the median of min(life, T) is
    min(tau * ln 2, T), which is a very different number from the mean
    tau * (1 - exp(-T/tau)): at tau = 1000 over 501 frames the median emitter never
    bleaches at all (live fraction 1.0) while the mean live fraction is 0.79.
    """
    live = pp.bleach_live_fraction_median(params.get("tau_bleach"), n_frames,
                                          params.get("bleach_shape", 1.0))
    return float(blink_duty(params) * live)


def bleach_tau_for_median_duty(params, target_duty, n_frames):
    """tau_bleach that gives the median emitter target_duty, keeping the blink ratio.

    Bleaching is the knob rather than the blink ratio because the blink ratio also
    sets how often off-times exceed SCP's 3-frame tracking gap, which is what gate
    T4 depends on; moving tau_bleach changes duty without disturbing that.

    Inverts median_duty(): need_live = target_duty / blink_duty, and
    tau = need_live * T / ln 2. Returns (tau, achievable); when need_live >= 1 the
    duty is unreachable by bleaching alone and the blink ratio has to give.
    """
    blink = blink_duty(params)
    need_live = target_duty / blink if blink > 0 else float("inf")
    if not np.isfinite(need_live) or need_live <= 0:
        return params.get("tau_bleach"), False
    if need_live >= 0.999:
        return 0.0, False        # 0 = never bleaches; duty is capped at `blink`
    return float(pp.bleach_tau_for_live_fraction(
        need_live, n_frames, params.get("bleach_shape", 1.0))), True


def duty_needed_from_measurement(duty_now, cv_now, cv_target):
    """Duty to aim for, calibrated on the measured CV rather than on the model.

    The two-level model gives CV^2 = k/duty - 1 for some constant k that collects
    the jitter width AND the measurement noise floor. Rather than model k, read it
    off the current measurement (k = duty_now * (1 + cv_now^2)) and solve for the
    duty that would give cv_target. That absorbs whatever the noise contributes and
    makes one iteration land close, instead of relying on the analytic k which
    ignores noise and is biased.
    """
    if not all(np.isfinite([duty_now, cv_now, cv_target])) or cv_target <= 0:
        return float("nan")
    k = duty_now * (1.0 + cv_now ** 2)
    return float(k / (1.0 + cv_target ** 2))


def measure_once(cfg, seed, movie_cache=None):
    """Generate one movie and return its photometry measurement."""
    movie, spots, meta = sm.simulate(cfg, seed=seed)
    targets = [(float(x), float(y)) for x, y in spots.get("target", [])]
    offs = [(float(x), float(y)) for x, y in spots.get("off_target", [])]
    ph = ms.photometry_report(movie, targets, offs)
    psf = ms.psf_report(movie, targets, offs)
    psf["target"]["fwhm_sigma"] = ms.fwhm_sigma_from_profile(
        psf["target"]["radial_profile"])
    if movie_cache is not None:
        movie_cache.update({"movie": movie, "spots": spots, "meta": meta})
    return ph, psf, meta


def report(cfg, ref, seeds, with_detector=False):
    """Measure the current config and print the section 4d comparison."""
    n_frames = int(cfg["global_settings"]["n_frames"])
    per_seed = []
    cache = {}
    for s in seeds:
        ph, psf, meta = measure_once(cfg, s, movie_cache=cache)
        per_seed.append((ph, psf, meta))

    def pooled(key, cls, field="median"):
        vals = np.concatenate([
            np.asarray([v for v in ph["per_spot_values"][key][cls]], dtype=np.float64)
            for ph, _, _ in per_seed])
        return ms.quartiles(vals)[field]

    print(f"\n  {'quantity':<34s} {'class':<11s} {'sim':>9s} {'real':>9s} {'ratio':>7s}")
    out = {}
    for key in ms.PHOTOMETRY_KEYS:
        for cls, refcls in CLASSES:
            sv = pooled(key, cls)
            rv = ref["photometry"][refcls][key]["median"]
            out[(key, cls)] = (sv, rv)
            flag = "" if (rv and abs(sv - rv) / abs(rv) <= 0.25) else "   <-- off"
            print(f"  {ms.PHOTOMETRY_LABELS[key]:<34s} {cls:<11s} {sv:9.3f} "
                  f"{rv:9.3f} {sv / rv if rv else float('nan'):7.3f}{flag}")
    sa = ms.auc(
        np.concatenate([np.asarray(ph["per_spot_values"]["bgsub_intensity"]["target"],
                                   dtype=np.float64) for ph, _, _ in per_seed]),
        np.concatenate([np.asarray(ph["per_spot_values"]["bgsub_intensity"]["off_target"],
                                   dtype=np.float64) for ph, _, _ in per_seed]))
    ra = ref["photometry"]["auc"]["bgsub_intensity"]
    print(f"  {'AUC of bg-subtracted brightness':<34s} {'-':<11s} {sa:9.3f} "
          f"{ra:9.3f}   (gate wants 0.85-0.95)")

    psf0 = per_seed[0][1]["target"]
    rpsf = ref.get("psf", {}).get("target", {})
    print(f"\n  PSF sigma: sim {psf0['stacked_second_moment_sigma']:.3f} "
          f"(2nd moment) / {psf0['fwhm_sigma']:.3f} (FWHM)   "
          f"real {rpsf.get('stacked_second_moment_sigma', float('nan')):.3f} / "
          f"{rpsf.get('fwhm_sigma', float('nan')):.3f}")

    print("\n  realised photophysics (duty of the MEDIAN emitter is what 4d reports)"
          f", mean of {len(per_seed)} seed(s):")
    for cls, _ in CLASSES:
        infos = [m["photophysics_realised"].get(cls) for _, _, m in per_seed]
        infos = [i for i in infos if i]
        if not infos:
            continue
        # Averaged over seeds, like the CV it is compared against. The median live
        # window of 243 gamma lifetimes has a standard error near 18 frames, so a
        # single seed is not a reliable read (320 vs 367 across two seeds).
        info = {k: (float(np.mean([i[k] for i in infos]))
                    if isinstance(infos[0][k], float) else infos[0][k])
                for k in infos[0] if k != "peak_amplitude_iqr"}
        cv_now = pooled("bgsub_cv", cls)
        cv_want = ref["photometry"][cls]["bgsub_cv"]["median"]
        want = duty_needed_from_measurement(info["overall_duty_median"], cv_now, cv_want)
        print(f"    {cls:11s} duty median {info['overall_duty_median']:.3f} "
              f"(mean {info['overall_duty_realised']:.3f}, analytic median "
              f"{median_duty(cfg[cls]['params'], n_frames):.3f})")
        print(f"    {'':11s} CV {cv_now:.3f} -> want {cv_want:.3f} => duty should be "
              f"{want:.3f};  live window median "
              f"{info['live_window_median_frames']:.0f} fr;  off-times past "
              f"tracker gap {info['off_time_fraction_exceeding_tracker_gap']:.3f}")

    if with_detector and cache:
        print("\n  track duration (one SCP run at the section 4f config) ...")
        tif = HERE / "_calibrate_probe.tif"
        import tifffile
        tifffile.imwrite(str(tif), cache["movie"], imagej=True, metadata={"axes": "TYX"})
        events, ok = ms.run_detector(str(tif), output_json=str(HERE / "_calibrate_probe.json"))
        if ok:
            targets = [(float(x), float(y)) for x, y in cache["spots"].get("target", [])]
            offs = [(float(x), float(y)) for x, y in cache["spots"].get("off_target", [])]
            td = ms.track_durations_by_class(events, targets, offs)
            scale = n_frames / ref["geometry"]["n_frames"]
            for cls in ("target", "off_target", "spurious"):
                rv = ref["track_duration"][cls]["median"] * scale
                sv = td[cls]["median"]
                hint = ""
                if np.isfinite(sv) and np.isfinite(rv) and rv:
                    if sv > rv * 1.3:
                        hint = "  -> too long: lower tau_bleach"
                    elif sv < rv * 0.7:
                        hint = "  -> too short: raise tau_bleach"
                print(f"    {cls:11s} sim {sv:7.1f}  real(scaled) {rv:7.1f}  "
                      f"n={td[cls]['n']}{hint}")
            perf = ms.pm.compute_metrics(targets, offs,
                                         [(float(e['x']), float(e['y'])) for e in events])
            print(f"    reference detector on this movie: R {perf['recall']:.1%} "
                  f"P {perf['precision']:.1%} F1 {perf['fbeta']:.3f}")
        else:
            print("    detector run failed")

    cv = {cls: pooled("bgsub_cv", cls) for cls, _ in CLASSES}
    # Averaged across seeds, to match the pooled CV it gets paired with. Reading
    # the duty off seed 0 alone while the CV is pooled is what stalled a first
    # calibration attempt: the median live window of 243 gamma-distributed
    # lifetimes has a standard error near 18 frames, so seed 0 gave 320 frames
    # where seed 1 gave 367, and the correction chased that noise instead of
    # converging.
    duty = {cls: float(np.mean([
        m["photophysics_realised"].get(cls, {}).get("overall_duty_median", np.nan)
        for _, _, m in per_seed])) for cls, _ in CLASSES}
    iqr = {cls: {"q1": pooled("bgsub_intensity", cls, "q1"),
                 "q3": pooled("bgsub_intensity", cls, "q3")}
           for cls, _ in CLASSES}
    print("\n  bg-subtracted brightness IQR (sets the AUC once medians match):")
    for cls, refcls in CLASSES:
        r = ref["photometry"][refcls]["bgsub_intensity"]
        s = iqr[cls]
        print(f"    {cls:11s} sim [{s['q1']:7.2f}, {s['q3']:7.2f}] ratio "
              f"{s['q3'] / s['q1'] if s['q1'] > 0 else float('nan'):5.2f}   "
              f"real [{r['q1']:7.2f}, {r['q3']:7.2f}] ratio "
              f"{r['q3'] / r['q1'] if r['q1'] > 0 else float('nan'):5.2f}")
    return out, sa, cv, duty, iqr


def fit(cfg, ref, seeds, fit_amplitudes=True, fit_duty=True, iterations=2):
    """Iteratively rescale amplitudes and retarget duty cycles."""
    n_frames = int(cfg["global_settings"]["n_frames"])
    for it in range(iterations):
        print(f"\n{'=' * 78}\nFIT ITERATION {it + 1}/{iterations}\n{'=' * 78}")
        out, _, out_cv, duty_now, out_iqr = report(cfg, ref, seeds)

        if fit_duty:
            print("\n  retargeting duty cycles from the measured CV ...")
            for cls, refcls in CLASSES:
                p = cfg[cls]["params"]
                cv_now, cv_want = out_cv[cls], ref["photometry"][refcls]["bgsub_cv"]["median"]
                want = duty_needed_from_measurement(duty_now[cls], cv_now, cv_want)
                if not np.isfinite(want):
                    continue
                tb, ok = bleach_tau_for_median_duty(p, want, n_frames)
                if not ok:
                    # Duty above the blink cap: bleaching cannot raise it, so the
                    # blink ratio has to. Lengthen tau_on, keeping tau_off (and so
                    # the tracker-gap fraction that T4 depends on) untouched.
                    cap = blink_duty(p)
                    if want > cap:
                        tau_off = p.get("tau_off") or 1.0
                        new_on = tau_off * want / max(1e-6, 1.0 - want)
                        print(f"    {cls:11s} duty {want:.3f} exceeds the blink cap "
                              f"{cap:.3f}; tau_on {p.get('tau_on')} -> {new_on:.2f} "
                              f"and tau_bleach -> 0 (never bleaches)")
                        p["tau_on"] = round(float(new_on), 2)
                        p["tau_bleach"] = 0.0
                    continue
                print(f"    {cls:11s} CV {cv_now:.3f} -> {cv_want:.3f}: duty "
                      f"{duty_now[cls]:.3f} -> {want:.3f}, tau_bleach "
                      f"{p.get('tau_bleach')} -> {tb:.1f}")
                p["tau_bleach"] = round(float(tb), 2)

        if fit_amplitudes:
            print("\n  rescaling peak amplitudes (bg-subtracted brightness is linear) ...")
            # Re-measure after the duty change, since duty scales brightness too.
            if fit_duty:
                out, _, out_cv, duty_now, out_iqr = report(cfg, ref, seeds)
            for cls, refcls in CLASSES:
                sv, rv = out[("bgsub_intensity", cls)]
                if not (np.isfinite(sv) and sv > 0 and np.isfinite(rv)):
                    continue
                p = cfg[cls]["params"]
                old = float(p["peak_amplitude_median"])
                new = old * (rv / sv)
                print(f"    {cls:11s} peak_amplitude_median {old:.1f} -> {new:.1f} "
                      f"(brightness {sv:.2f} -> target {rv:.2f})")
                p["peak_amplitude_median"] = round(new, 2)

            # Match the SPREAD, not only the centre. This is what gate T3's AUC
            # criterion actually tests: with the medians already matched, the AUC of
            # bg-subtracted brightness is set entirely by how much the two classes'
            # distributions overlap.
            #
            # The correction has to be multiplicative on the MEASURED spread, not a
            # move toward the spread implied by the real IQR. The measured spread
            # contains two contributions -- the lognormal amplitude AND the spread
            # in per-emitter duty that bleaching creates -- so setting sigma_log to
            # the value the real IQR implies double-counts, and leaves the simulated
            # distributions too wide. Measured: doing that gave an AUC of 0.780
            # against the real 0.909, with every median already matched to 0.3%.
            for cls, refcls in CLASSES:
                p = cfg[cls]["params"]
                r = ref["photometry"][refcls]["bgsub_intensity"]
                s = out_iqr[cls]
                if not (r["q1"] > 0 and r["q3"] > 0 and s["q1"] > 0 and s["q3"] > 0):
                    continue
                want_ratio = r["q3"] / r["q1"]
                have_ratio = s["q3"] / s["q1"]
                if have_ratio <= 1.0:
                    continue
                # log(IQR ratio) is proportional to the total log-spread, so scale
                # sigma_log by the ratio of the wanted to the achieved log-spread.
                old = float(p["peak_amplitude_sigma_log"])
                new = max(0.02, old * np.log(want_ratio) / np.log(have_ratio))
                print(f"    {cls:11s} peak_amplitude_sigma_log {old:.3f} -> {new:.3f} "
                      f"(measured IQR ratio {have_ratio:.2f} -> want "
                      f"{want_ratio:.2f})")
                p["peak_amplitude_sigma_log"] = round(float(new), 3)
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--seeds", default="0,1", help="comma-separated seeds to average")
    ap.add_argument("--fit-amplitudes", action="store_true")
    ap.add_argument("--fit-duty", action="store_true")
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--with-detector", action="store_true",
                    help="also report the T4 track-duration diagnostic (slow)")
    ap.add_argument("--write", action="store_true",
                    help="write the fitted config back to --config")
    args = ap.parse_args()

    with open(args.reference, encoding="utf-8") as f:
        ref = json.load(f)
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = raw["simulation"]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    print("=" * 78)
    print("CALIBRATING AGAINST THE REAL MOVIE'S PHOTOMETRY")
    print("=" * 78)
    print(f"  config   : {args.config}")
    print(f"  reference: {args.reference}")
    print(f"  seeds    : {seeds}")

    if args.fit_amplitudes or args.fit_duty:
        cfg = fit(cfg, ref, seeds, fit_amplitudes=args.fit_amplitudes,
                  fit_duty=args.fit_duty, iterations=args.iterations)
        print(f"\n{'=' * 78}\nAFTER FITTING\n{'=' * 78}")

    report(cfg, ref, seeds, with_detector=args.with_detector)

    if args.write:
        raw["simulation"] = cfg
        # Round-trip through yaml, losing the comments -- so write to a sidecar and
        # let a human merge, rather than silently destroying the documented config.
        out = Path(args.config).with_suffix(".fitted.yaml")
        out.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        print(f"\nFitted values written to {out}")
        print("Merge them into the documented config by hand: yaml.safe_dump drops "
              "the comments,\nand every value in real_calibrated.yaml carries the "
              "measurement that justifies it.")


if __name__ == "__main__":
    main()
