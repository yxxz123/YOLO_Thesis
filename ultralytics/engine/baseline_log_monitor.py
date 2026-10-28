# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAP50_METRIC = "metrics/mAP50(B)"
RECALL_METRIC = "metrics/recall(B)"


@dataclass
class BaselineLogDecision:
    active: bool
    should_stop: bool
    epoch: int
    metric: str

    current: float | None = None
    baseline: float | None = None
    diff: float | None = None

    # 5-epoch rolling comparison
    rolling_current: float | None = None
    rolling_baseline: float | None = None
    rolling_diff: float | None = None

    # recent 20-epoch best comparison
    recent_best_current: float | None = None
    recent_best_baseline: float | None = None
    recent_best_diff: float | None = None

    # auxiliary metrics
    recall_diff: float | None = None
    map50_diff: float | None = None

    # weighted score
    score_diff: float | None = None

    bad_epochs: int = 0
    patience: int = 1

    catastrophic_bad_epochs: int = 0

    reason: str = ""


class BaselineLogMonitor:
    """
    Automatic candidate-model screening against a fixed DySample baseline.

    Screening strategy
    ------------------
    1. Catastrophic failure detection:
       Kill obviously collapsed models early.

    2. Regular screening:
       Starting around 45% of total epochs, compare 5-epoch moving averages
       instead of single-epoch values.

    3. Recent-best protection:
       A candidate is not eliminated only because several individual epochs
       fluctuate badly if its recent 20-epoch best remains competitive.

    4. Auxiliary protection:
       Recall improvement can protect a candidate even if mAP50-95 is
       temporarily slightly worse.

    5. Late hard gate:
       Around 67% of training, a candidate that is still clearly behind both
       in rolling average and recent best is stopped.
    """

    def __init__(
        self,
        baseline_by_epoch: dict[int, dict[str, float]],
        metric: str,
        start_epoch: int,
        min_gap: float,
        patience: int,
        total_epochs: int,
    ):
        self.baseline_by_epoch = baseline_by_epoch
        self.metric = metric
        self.start_epoch = max(1, int(start_epoch))
        self.min_gap = float(min_gap)
        self.patience = max(1, int(patience))
        self.total_epochs = int(total_epochs)

        self.bad_epochs = 0
        self.catastrophic_bad_epochs = 0

        # Store candidate validation metrics by epoch
        self.current_by_epoch: dict[int, dict[str, float]] = {}

        # =====================================================
        # Screening policy
        # =====================================================

        # Main rolling comparison
        self.rolling_window = 5

        # Protect candidates with a good recent peak
        self.recent_best_window = 20

        # Recent-best must also lag baseline by at least 0.3 pt
        # before a normal elimination is allowed.
        self.recent_best_min_gap = 0.003

        # -----------------------------------------------------
        # Catastrophic failure detection
        # -----------------------------------------------------

        # Do not evaluate catastrophic collapse too early.
        self.catastrophic_start_epoch = max(
            20,
            int(self.total_epochs * 0.13),
        )

        # Candidate rolling mAP50-95 must lag >10 percentage points
        # AND be below 50% of baseline performance.
        self.catastrophic_gap = 0.10
        self.catastrophic_ratio = 0.50
        self.catastrophic_patience = 2
        self.catastrophic_window = 3

        # -----------------------------------------------------
        # Late-stage hard gate
        # -----------------------------------------------------

        # 150 epochs -> around epoch 100
        self.stage_gate_epoch = max(
            self.start_epoch,
            int(self.total_epochs * 0.67),
        )

        # At epoch ~100, if both rolling average and recent-best
        # are still >0.5 pt behind baseline, stop.
        self.stage_gate_gap = 0.005

        # -----------------------------------------------------
        # Auxiliary-metric protection
        # -----------------------------------------------------

        # If recall improves by at least +1.0 pt, preserve candidate
        # even when mAP50-95 is temporarily worse.
        self.recall_protection_gain = 0.010

        # Composite comparison:
        #
        # 70% mAP50-95
        # 20% Recall
        # 10% mAP50
        #
        self.weight_map95 = 0.70
        self.weight_recall = 0.20
        self.weight_map50 = 0.10

    # =========================================================
    # Construction
    # =========================================================

    @classmethod
    def from_args(
        cls,
        enabled: bool,
        path: str | None,
        metric: str,
        total_epochs: int,
        start_ratio: float,
        min_gap: float,
        patience: int,
    ) -> "BaselineLogMonitor | None":

        if not enabled:
            return None

        if not path:
            raise ValueError(
                "baseline_path is required when baseline_monitor=True"
            )

        start_epoch = max(
            1,
            int(math.floor(int(total_epochs) * float(start_ratio))),
        )

        return cls(
            baseline_by_epoch=cls._load_results_csv(
                path,
                primary_metric=metric,
            ),
            metric=metric,
            start_epoch=start_epoch,
            min_gap=min_gap,
            patience=patience,
            total_epochs=total_epochs,
        )

    # =========================================================
    # CSV loading
    # =========================================================

    @staticmethod
    def _load_results_csv(
        path: str | Path,
        primary_metric: str,
    ) -> dict[int, dict[str, float]]:

        log_path = Path(path)

        if not log_path.exists():
            raise FileNotFoundError(
                f"Baseline results.csv does not exist: {log_path}"
            )

        metrics_to_load = {
            primary_metric,
            MAP50_METRIC,
            RECALL_METRIC,
        }

        baseline_by_epoch: dict[int, dict[str, float]] = {}

        with log_path.open(newline="", encoding="utf-8") as f:

            reader = csv.DictReader(
                f,
                skipinitialspace=True,
            )

            if reader.fieldnames:
                reader.fieldnames = [
                    field.strip()
                    for field in reader.fieldnames
                ]

            for row in reader:

                normalized = {
                    key.strip(): value
                    for key, value in row.items()
                    if key is not None
                }

                epoch = BaselineLogMonitor._to_int(
                    normalized.get("epoch")
                )

                if epoch is None:
                    continue

                record: dict[str, float] = {}

                for metric_name in metrics_to_load:

                    value = BaselineLogMonitor._to_float(
                        normalized.get(metric_name)
                    )

                    if value is not None:
                        record[metric_name] = value

                baseline_by_epoch[epoch] = record

        return baseline_by_epoch

    # =========================================================
    # Helpers
    # =========================================================

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value: Any) -> int | None:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _record_current(
        self,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:

        record: dict[str, float] = {}

        for metric_name in {
            self.metric,
            MAP50_METRIC,
            RECALL_METRIC,
        }:

            value = self._to_float(
                metrics.get(metric_name)
            )

            if value is not None:
                record[metric_name] = value

        self.current_by_epoch[epoch] = record

    @staticmethod
    def _window_mean(
        records: dict[int, dict[str, float]],
        metric: str,
        end_epoch: int,
        window: int,
    ) -> float | None:

        start_epoch = max(
            1,
            end_epoch - window + 1,
        )

        values: list[float] = []

        for epoch in range(
            start_epoch,
            end_epoch + 1,
        ):

            record = records.get(epoch)

            if record is None:
                continue

            value = record.get(metric)

            if value is not None:
                values.append(value)

        # Require the full window.
        if len(values) < window:
            return None

        return sum(values) / len(values)

    @staticmethod
    def _window_best(
        records: dict[int, dict[str, float]],
        metric: str,
        end_epoch: int,
        window: int,
    ) -> float | None:

        start_epoch = max(
            1,
            end_epoch - window + 1,
        )

        values: list[float] = []

        for epoch in range(
            start_epoch,
            end_epoch + 1,
        ):

            record = records.get(epoch)

            if record is None:
                continue

            value = record.get(metric)

            if value is not None:
                values.append(value)

        if not values:
            return None

        return max(values)

    # =========================================================
    # Screening
    # =========================================================

    def check(
        self,
        metrics: dict[str, Any] | None,
        epoch: int,
    ) -> BaselineLogDecision:

        epoch = int(epoch)
        metrics = metrics or {}

        self._record_current(
            epoch,
            metrics,
        )

        baseline_record = self.baseline_by_epoch.get(
            epoch
        )

        current = self._to_float(
            metrics.get(self.metric)
        )

        baseline = (
            baseline_record.get(self.metric)
            if baseline_record
            else None
        )

        diff = (
            current - baseline
            if current is not None
            and baseline is not None
            else None
        )

        # =====================================================
        # Rolling mAP50-95
        # =====================================================

        rolling_current = self._window_mean(
            self.current_by_epoch,
            self.metric,
            epoch,
            self.rolling_window,
        )

        rolling_baseline = self._window_mean(
            self.baseline_by_epoch,
            self.metric,
            epoch,
            self.rolling_window,
        )

        rolling_diff = (
            rolling_current - rolling_baseline
            if rolling_current is not None
            and rolling_baseline is not None
            else None
        )

        # =====================================================
        # Recent best
        # =====================================================

        recent_best_current = self._window_best(
            self.current_by_epoch,
            self.metric,
            epoch,
            self.recent_best_window,
        )

        recent_best_baseline = self._window_best(
            self.baseline_by_epoch,
            self.metric,
            epoch,
            self.recent_best_window,
        )

        recent_best_diff = (
            recent_best_current - recent_best_baseline
            if recent_best_current is not None
            and recent_best_baseline is not None
            else None
        )

        # =====================================================
        # Recall rolling comparison
        # =====================================================

        recall_current = self._window_mean(
            self.current_by_epoch,
            RECALL_METRIC,
            epoch,
            self.rolling_window,
        )

        recall_baseline = self._window_mean(
            self.baseline_by_epoch,
            RECALL_METRIC,
            epoch,
            self.rolling_window,
        )

        recall_diff = (
            recall_current - recall_baseline
            if recall_current is not None
            and recall_baseline is not None
            else None
        )

        # =====================================================
        # mAP50 rolling comparison
        # =====================================================

        map50_current = self._window_mean(
            self.current_by_epoch,
            MAP50_METRIC,
            epoch,
            self.rolling_window,
        )

        map50_baseline = self._window_mean(
            self.baseline_by_epoch,
            MAP50_METRIC,
            epoch,
            self.rolling_window,
        )

        map50_diff = (
            map50_current - map50_baseline
            if map50_current is not None
            and map50_baseline is not None
            else None
        )

        # =====================================================
        # Composite score
        # =====================================================

        score_diff = None

        if rolling_diff is not None:

            r_gap = (
                recall_diff
                if recall_diff is not None
                else 0.0
            )

            m50_gap = (
                map50_diff
                if map50_diff is not None
                else 0.0
            )

            score_diff = (
                self.weight_map95 * rolling_diff
                + self.weight_recall * r_gap
                + self.weight_map50 * m50_gap
            )

        # =====================================================
        # Layer 1:
        # Catastrophic failure
        # =====================================================

        if epoch >= self.catastrophic_start_epoch:

            cat_current = self._window_mean(
                self.current_by_epoch,
                self.metric,
                epoch,
                self.catastrophic_window,
            )

            cat_baseline = self._window_mean(
                self.baseline_by_epoch,
                self.metric,
                epoch,
                self.catastrophic_window,
            )

            catastrophic = False

            if (
                cat_current is not None
                and cat_baseline is not None
                and cat_baseline > 0
            ):

                catastrophic = (
                    (cat_baseline - cat_current)
                    > self.catastrophic_gap
                    and
                    cat_current
                    < cat_baseline
                    * self.catastrophic_ratio
                )

            if catastrophic:
                self.catastrophic_bad_epochs += 1
            else:
                self.catastrophic_bad_epochs = 0

            if (
                self.catastrophic_bad_epochs
                >= self.catastrophic_patience
            ):

                return BaselineLogDecision(
                    active=True,
                    should_stop=True,
                    epoch=epoch,
                    metric=self.metric,
                    current=current,
                    baseline=baseline,
                    diff=diff,
                    rolling_current=rolling_current,
                    rolling_baseline=rolling_baseline,
                    rolling_diff=rolling_diff,
                    recent_best_current=recent_best_current,
                    recent_best_baseline=recent_best_baseline,
                    recent_best_diff=recent_best_diff,
                    recall_diff=recall_diff,
                    map50_diff=map50_diff,
                    score_diff=score_diff,
                    bad_epochs=self.bad_epochs,
                    patience=self.patience,
                    catastrophic_bad_epochs=(
                        self.catastrophic_bad_epochs
                    ),
                    reason=(
                        "catastrophic performance collapse: "
                        f"{self.catastrophic_bad_epochs} "
                        "consecutive checks"
                    ),
                )

        # =====================================================
        # Wait for normal screening start
        # =====================================================

        if epoch < self.start_epoch:

            return BaselineLogDecision(
                active=False,
                should_stop=False,
                epoch=epoch,
                metric=self.metric,
                current=current,
                baseline=baseline,
                diff=diff,
                rolling_current=rolling_current,
                rolling_baseline=rolling_baseline,
                rolling_diff=rolling_diff,
                recent_best_current=recent_best_current,
                recent_best_baseline=recent_best_baseline,
                recent_best_diff=recent_best_diff,
                recall_diff=recall_diff,
                map50_diff=map50_diff,
                score_diff=score_diff,
                bad_epochs=self.bad_epochs,
                patience=self.patience,
                catastrophic_bad_epochs=(
                    self.catastrophic_bad_epochs
                ),
                reason=(
                    f"waiting for start_epoch="
                    f"{self.start_epoch}"
                ),
            )

        # Missing data -> do not stop.
        if (
            rolling_diff is None
            or recent_best_diff is None
        ):

            return BaselineLogDecision(
                active=True,
                should_stop=False,
                epoch=epoch,
                metric=self.metric,
                current=current,
                baseline=baseline,
                diff=diff,
                rolling_current=rolling_current,
                rolling_baseline=rolling_baseline,
                rolling_diff=rolling_diff,
                recent_best_current=recent_best_current,
                recent_best_baseline=recent_best_baseline,
                recent_best_diff=recent_best_diff,
                recall_diff=recall_diff,
                map50_diff=map50_diff,
                score_diff=score_diff,
                bad_epochs=self.bad_epochs,
                patience=self.patience,
                catastrophic_bad_epochs=(
                    self.catastrophic_bad_epochs
                ),
                reason="insufficient rolling data",
            )

        # =====================================================
        # Recall protection
        # =====================================================

        recall_protected = (
            recall_diff is not None
            and recall_diff >= self.recall_protection_gain

            # Recall gain cannot compensate for a large
            # mAP50-95 degradation.
            and rolling_diff is not None
            and rolling_diff >= -0.005

            # Recent best must also remain reasonably competitive.
            and recent_best_diff is not None
            and recent_best_diff >= -0.005
)

        # =====================================================
        # Layer 2:
        # Normal rolling elimination
        # =====================================================

        normal_bad = (
            rolling_diff < -self.min_gap

            # Candidate's recent best must also be clearly worse.
            and recent_best_diff
            < -self.recent_best_min_gap

            # Composite score confirms overall degradation.
            and (
                score_diff is None
                or score_diff
                < -(self.min_gap * 0.75)
            )

            # Do not eliminate a meaningful Recall improvement.
            and not recall_protected
        )

        if normal_bad:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0

        # =====================================================
        # Layer 3:
        # Late-stage hard gate
        # =====================================================

        late_stage_bad = False

        if epoch >= self.stage_gate_epoch:

            late_stage_bad = (
                rolling_diff
                < -self.stage_gate_gap

                and recent_best_diff
                < -self.stage_gate_gap

                and not recall_protected
            )

        if late_stage_bad:

            return BaselineLogDecision(
                active=True,
                should_stop=True,
                epoch=epoch,
                metric=self.metric,
                current=current,
                baseline=baseline,
                diff=diff,
                rolling_current=rolling_current,
                rolling_baseline=rolling_baseline,
                rolling_diff=rolling_diff,
                recent_best_current=recent_best_current,
                recent_best_baseline=recent_best_baseline,
                recent_best_diff=recent_best_diff,
                recall_diff=recall_diff,
                map50_diff=map50_diff,
                score_diff=score_diff,
                bad_epochs=self.bad_epochs,
                patience=self.patience,
                catastrophic_bad_epochs=(
                    self.catastrophic_bad_epochs
                ),
                reason=(
                    "late-stage hard gate: candidate remains "
                    "clearly below baseline in both rolling "
                    "average and recent-best performance"
                ),
            )

        # =====================================================
        # Normal patience-based stop
        # =====================================================

        should_stop = (
            self.bad_epochs
            >= self.patience
        )

        reason = ""

        if should_stop:
            reason = (
                "candidate underperforms baseline after "
                "rolling-average and recent-best checks: "
                f"{self.bad_epochs} consecutive bad epochs"
            )

        return BaselineLogDecision(
            active=True,
            should_stop=should_stop,
            epoch=epoch,
            metric=self.metric,
            current=current,
            baseline=baseline,
            diff=diff,
            rolling_current=rolling_current,
            rolling_baseline=rolling_baseline,
            rolling_diff=rolling_diff,
            recent_best_current=recent_best_current,
            recent_best_baseline=recent_best_baseline,
            recent_best_diff=recent_best_diff,
            recall_diff=recall_diff,
            map50_diff=map50_diff,
            score_diff=score_diff,
            bad_epochs=self.bad_epochs,
            patience=self.patience,
            catastrophic_bad_epochs=(
                self.catastrophic_bad_epochs
            ),
            reason=reason,
        )