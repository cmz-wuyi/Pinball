"""
12口弹珠机最佳出卡策略模块
Optimal strategy module for the 12-slot pinball machine.

Rules:
- The machine requires a minimum of MIN_BET (5) small marbles to start each play.
- After pressing the button the multiplier is revealed (2x/4x/6x/8x/10x).
- Typical slot distribution per multiplier:
    2x  -> 4 slots lit   (P_win ≈ 4/12)
    4x  -> 3 slots lit   (P_win ≈ 3/12)
    6x  -> 2 slots lit   (P_win ≈ 2/12)
    8x  -> 1 slot lit    (P_win ≈ 1/12)
    10x -> 1 slot lit    (P_win ≈ 1/12)
- Before shooting you may add more marbles up to a total of 99.
- A win returns (multiplier × total_bet) marbles and score cards based on card tiers.
"""

import math
from typing import Dict, List, Optional, Tuple

# Number of physical slots on the machine
NUM_SLOTS = 12

# Absolute minimum marbles required to start a play
MIN_BET = 5

# Maximum total marbles per play
MAX_BET = 99

# Default multiplier -> number of lit slots mapping
DEFAULT_MULTIPLIER_SLOTS: Dict[int, int] = {
    2: 4,
    4: 3,
    6: 2,
    8: 1,
    10: 1,
}

# Default theoretical multiplier probabilities (from README)
DEFAULT_MULTIPLIER_PROBS: Dict[int, float] = {
    2: 0.420,
    4: 0.288,
    6: 0.127,
    8: 0.108,
    10: 0.057,
}


def calculate_score_cards(returned_marbles: int, card_tiers: List[Tuple[int, int]]) -> int:
    """
    Calculate score cards based on tiered thresholds.
    
    Args:
        returned_marbles: Total marbles returned (multiplier × bet)
        card_tiers: List of (threshold, cards) tuples, sorted by threshold ascending
    
    Returns:
        Number of score cards earned
    """
    cards = 0
    for threshold, reward in card_tiers:
        if returned_marbles >= threshold:
            cards = reward
        else:
            break
    return cards


def make_card_tiers(T: int, J: int) -> List[Tuple[int, int]]:
    """
    Create standard tiered card rules from legacy T and J parameters.
    
    Standard rule: every T marbles earns 1 card, up to J cards max.
    e.g., T=20, J=3 -> [(20,1), (40,2), (60,3)]
    
    Args:
        T: Base threshold for 1 card
        J: Maximum cards per win
    
    Returns:
        List of (threshold, cards) tuples
    """
    tiers = []
    for k in range(1, J + 1):
        tiers.append((T * k, k))
    return tiers


class PinballStrategy:
    """
    Strategy advisor for the 12-slot pinball machine.

    It maintains a running estimate of the physical landing probability of the
    marble across all 12 slots and uses that estimate—together with the current
    multiplier and the set of lit slots—to recommend the optimal number of
    marbles to bet each round.

    Parameters
    ----------
    card_tiers : list of tuples, optional
        Custom tiered card rules: list of (threshold, cards) tuples.
        e.g., [(100, 1), (150, 2), (200, 3)] means 100返珠=1卡, 150返珠=2卡, 200返珠=3卡.
        If not provided, uses standard T/J rules.
    T : int, optional
        Score-card divisor (legacy): every T returned marbles yield one score card.
        Used only if card_tiers is not provided.
    J : int, optional
        Maximum number of score cards (legacy). Used only if card_tiers is not provided.
    priority : str
        ``'cards'``   – maximise score-card yield per marble spent.
        ``'marbles'`` – maximise marble return (minimise losses / maximise EV).
    multiplier_slots : dict, optional
        Mapping of multiplier value to the number of lit slots.  Defaults to
        ``DEFAULT_MULTIPLIER_SLOTS``.
    current_marbles : int, optional
        Current number of marbles the player has. Used to limit bet recommendations.
    """

    def __init__(
        self,
        T: int = 20,
        J: int = 10,
        priority: str = "cards",
        multiplier_slots: Optional[Dict[int, int]] = None,
        prior_weight: float = 24.0,
        confidence_threshold: float = 0.0,
        max_bet: int = MAX_BET,
        card_tiers: Optional[List[Tuple[int, int]]] = None,
        current_marbles: int = 999,
    ) -> None:
        if priority not in ("cards", "marbles"):
            raise ValueError("priority must be 'cards' or 'marbles'")
        if prior_weight < 0:
            raise ValueError(f"prior_weight must be >= 0, got {prior_weight}")
        if confidence_threshold < 0:
            raise ValueError(f"confidence_threshold must be >= 0, got {confidence_threshold}")
        if not (MIN_BET <= max_bet <= MAX_BET):
            raise ValueError(f"max_bet must be between {MIN_BET} and {MAX_BET}, got {max_bet}")

        self.priority = priority
        self.max_bet = max_bet
        self.multiplier_slots: Dict[int, int] = (
            multiplier_slots if multiplier_slots is not None else dict(DEFAULT_MULTIPLIER_SLOTS)
        )
        self.prior_weight = prior_weight
        self.confidence_threshold = confidence_threshold
        self.current_marbles = current_marbles

        # Card tier configuration
        if card_tiers is not None:
            # Validate custom card tiers
            self.card_tiers = sorted(card_tiers, key=lambda x: x[0])
            self.T = self.card_tiers[0][0] if self.card_tiers else 20
            self.J = self.card_tiers[-1][1] if self.card_tiers else 10
        else:
            # Legacy T/J parameters
            if T < 1:
                raise ValueError(f"T must be at least 1, got {T}")
            if J < 1:
                raise ValueError(f"J must be at least 1, got {J}")
            self.T = T
            self.J = J
            self.card_tiers = make_card_tiers(T, J)

        # Landing history: count of times the marble landed in each slot
        self._landing_counts: List[int] = [0] * NUM_SLOTS
        self._total_plays: int = 0
        
        # Multiplier occurrence tracking (for analysis)
        self._multiplier_counts: Dict[int, int] = {m: 0 for m in DEFAULT_MULTIPLIER_SLOTS.keys()}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def total_plays(self) -> int:
        """Total number of plays recorded so far."""
        return self._total_plays

    def get_landing_probs(self) -> List[float]:
        """
        Return the estimated probability distribution for where the marble lands.

        Uses Bayesian smoothing (Dirichlet prior) so that a small number of
        observations does not dominate the estimate.  The prior assumes a
        uniform distribution with total weight ``prior_weight`` (split equally
        across all slots).  With ``prior_weight=0`` this reduces to pure
        frequency counting.

        .. math::

            P(\\text{slot}_i)
            = \\frac{\\alpha + \\text{count}_i}
                   {N_{\\alpha} + \\text{total\\_plays}}

        where :math:`\\alpha = \\text{prior\\_weight} / N_{\\text{slots}}`.
        """
        alpha = self.prior_weight / NUM_SLOTS
        total = self.prior_weight + self._total_plays
        if total == 0:
            return [1.0 / NUM_SLOTS] * NUM_SLOTS
        return [(alpha + c) / total for c in self._landing_counts]

    def record_landing(self, slot: int, multiplier: Optional[int] = None) -> None:
        """
        Record which slot the marble landed in after a play.

        Parameters
        ----------
        slot : int
            0-indexed slot number (0 to NUM_SLOTS-1).
        multiplier : int, optional
            The multiplier for this play (for tracking distribution)
        """
        if not (0 <= slot < NUM_SLOTS):
            raise ValueError(f"slot must be between 0 and {NUM_SLOTS - 1}, got {slot}")
        self._landing_counts[slot] += 1
        self._total_plays += 1
        if multiplier is not None and multiplier in self._multiplier_counts:
            self._multiplier_counts[multiplier] += 1

    def update_marbles(self, delta: int) -> None:
        """
        Update the current marble count.

        Parameters
        ----------
        delta : int
            Change in marbles (positive for gain, negative for loss)
        """
        self.current_marbles = max(0, self.current_marbles + delta)

    # ------------------------------------------------------------------
    # Core strategy calculations
    # ------------------------------------------------------------------

    def win_probability(self, lit_slots: List[int]) -> float:
        """
        Estimate the probability of winning given a set of lit slots.

        Parameters
        ----------
        lit_slots : list of int
            0-indexed indices of the currently lit slots.
        """
        probs = self.get_landing_probs()
        unique_slots = set(s for s in lit_slots if 0 <= s < NUM_SLOTS)
        return sum(probs[s] for s in unique_slots)

    def optimal_bet(self, multiplier: int, lit_slots: List[int]) -> int:
        """
        Recommend the optimal total number of marbles to bet this round.

        Parameters
        ----------
        multiplier : int
            Reward multiplier shown after pressing the button.
        lit_slots : list of int
            0-indexed indices of the currently lit slots.

        Returns
        -------
        int
            Total marbles to commit (between MIN_BET and MAX_BET inclusive).
        """
        p_win = self.win_probability(lit_slots)

        if self.confidence_threshold > 0:
            # Adaptive mode: scale bet based on observation confidence
            if self.priority == "marbles":
                return self._bet_for_marbles_adaptive(multiplier, p_win)
            return self._bet_for_cards_adaptive(multiplier, p_win)

        if self.priority == "marbles":
            return self._bet_for_marbles(multiplier, p_win)
        return self._bet_for_cards(multiplier)

    def recommend(self, multiplier: int, lit_slots: List[int]) -> dict:
        """
        Return a full recommendation dict for the current play.

        Parameters
        ----------
        multiplier : int
            Reward multiplier shown after pressing the button.
        lit_slots : list of int
            0-indexed indices of the currently lit slots.

        Returns
        -------
        dict with keys:
            multiplier, lit_slots, win_probability, optimal_bet,
            expected_marble_return, expected_score_cards, marble_roi,
            max_possible_bet, card_tiers_info
        """
        p_win = self.win_probability(lit_slots)
        bet = self.optimal_bet(multiplier, lit_slots)
        expected_marbles = multiplier * bet * p_win
        returned_marbles_if_win = multiplier * bet
        expected_cards = p_win * calculate_score_cards(returned_marbles_if_win, self.card_tiers)
        expected_marbles_rounded = round(expected_marbles, 2)
        roi = expected_marbles_rounded / bet if bet > 0 else 0.0
        
        max_possible_bet = min(self.max_bet, self.current_marbles)

        return {
            "multiplier": multiplier,
            "lit_slots": lit_slots,
            "win_probability": round(p_win, 4),
            "optimal_bet": bet,
            "max_possible_bet": max_possible_bet,
            "expected_marble_return": expected_marbles_rounded,
            "expected_score_cards": round(expected_cards, 4),
            "marble_roi": round(roi, 4),
            "card_tiers": [[t[0], t[1]] for t in self.card_tiers],
            "current_marbles": self.current_marbles,
        }

    # ------------------------------------------------------------------
    # Analytical helpers (static / no instance state needed)
    # ------------------------------------------------------------------

    @staticmethod
    def expected_value_table(
        T: int,
        J: int,
        multiplier_slots: Optional[Dict[int, int]] = None,
        max_bet: int = MAX_BET,
    ) -> List[dict]:
        """
        Build an expected-value analysis table for each multiplier,
        assuming a uniform landing distribution (i.e., no historical data).

        Useful for understanding the baseline characteristics of the machine.
        """
        mb = max(MIN_BET, min(MAX_BET, max_bet))
        ms = multiplier_slots if multiplier_slots is not None else DEFAULT_MULTIPLIER_SLOTS
        rows = []
        for mult, n_lit in sorted(ms.items()):
            p_win = n_lit / NUM_SLOTS

            # --- Marble priority: bet MIN_BET (minimum) ---
            ev_marbles_per_marble = mult * p_win

            # --- Card priority: find the bet with best cards-per-marble ---
            n_card = max(MIN_BET, min(mb, math.ceil(T * J / mult)))
            best_eff = 0.0
            for k in range(1, J + 1):
                n = max(MIN_BET, math.ceil(k * T / mult))
                if n > mb:
                    break
                eff = k / n
                if eff > best_eff:
                    best_eff = eff
                    n_card = n
            cards_per_marble = p_win * min(mult * n_card // T, J) / n_card if n_card else 0

            rows.append(
                {
                    "multiplier": mult,
                    "lit_slots": n_lit,
                    "p_win": round(p_win, 4),
                    "marble_ev_ratio": round(ev_marbles_per_marble, 4),
                    "card_optimal_bet": n_card,
                    "cards_per_marble": round(cards_per_marble, 4),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bet_for_marbles(self, multiplier: int, p_win: float) -> int:
        """
        Marble-priority bet calculation.

        Bet the maximum (max_bet) only when the expected return exceeds the input.
        Otherwise bet the minimum (MIN_BET) to limit losses.
        """
        ev_ratio = multiplier * p_win
        if ev_ratio > 1.0:
            return self.max_bet
        return MIN_BET

    def _bet_for_cards(self, multiplier: int) -> int:
        """
        Card-priority bet calculation using tiered card rules.

        Finds the bet amount that maximizes cards-per-marble efficiency.
        Considers all card tiers defined in card_tiers.
        """
        available_bet = min(self.max_bet, self.current_marbles)
        best_n = MIN_BET
        best_eff = 0.0
        
        for threshold, cards in self.card_tiers:
            n = max(MIN_BET, math.ceil(threshold / multiplier))
            if n > available_bet:
                break
            eff = cards / n
            if eff > best_eff:
                best_eff = eff
                best_n = n
        
        return best_n

    # ------------------------------------------------------------------
    # Adaptive (confidence-aware) betting
    # ------------------------------------------------------------------

    def _confidence(self) -> float:
        """Return a 0→1 confidence measure based on observations so far."""
        return self._total_plays / (self._total_plays + self.confidence_threshold)

    def _bet_for_cards_adaptive(self, multiplier: int, p_win: float) -> int:
        """
        Confidence-aware card-priority bet with tiered card rules.

        Key insight: when mult × p_win < 1 (negative EV, which is most rounds),
        small step-aligned bets are equally or MORE card-efficient per marble
        than large bets, because large bets suffer floor-rounding waste.

        Strategy:
        - Negative EV: always bet the minimum step-aligned amount (1 card on win).
          This also provides cheap exploration data.
        - Positive EV detected: ramp toward available max bet based on confidence.
          Higher confidence → we trust the positive-EV signal more → bet bigger.
        """
        available_bet = min(self.max_bet, self.current_marbles)
        
        # Minimum bet that earns at least 1 card on a win
        n_floor = MIN_BET
        if self.card_tiers:
            min_threshold = self.card_tiers[0][0]
            n_floor = max(MIN_BET, math.ceil(min_threshold / multiplier))
        n_floor = min(n_floor, available_bet)

        ev_ratio = multiplier * p_win

        if ev_ratio <= 1.0:
            # Negative EV: small bet is most card-efficient per marble
            return n_floor

        # Positive EV: worth betting big (we gain marbles AND cards).
        # Scale with confidence to avoid overcommitting on noisy estimates.
        conf = self._confidence()
        bet = n_floor + round(conf * (available_bet - n_floor))
        return max(MIN_BET, min(available_bet, bet))

    def _bet_for_marbles_adaptive(self, multiplier: int, p_win: float) -> int:
        """
        Confidence-aware marble-priority bet.

        When confidence is low, require a stronger EV signal before betting big.
        As confidence grows, the threshold drops to the standard EV > 1.0.
        Also considers current marble count to avoid betting more than available.
        """
        ev_ratio = multiplier * p_win
        conf = self._confidence()
        available_bet = min(self.max_bet, self.current_marbles)

        # Required EV threshold: 1.5 at zero confidence → 1.0 at full confidence
        threshold = 1.0 + 0.5 * (1 - conf)

        if ev_ratio <= 1.0:
            return MIN_BET
        if ev_ratio >= threshold:
            return available_bet

        # Between 1.0 and threshold: scale proportionally
        fraction = (ev_ratio - 1.0) / max(0.01, threshold - 1.0) * conf
        return max(MIN_BET, min(available_bet, MIN_BET + round(fraction * (available_bet - MIN_BET))))

    # ------------------------------------------------------------------
    # Machine analysis (Chinese market optimizations)
    # ------------------------------------------------------------------

    def get_multiplier_distribution(self) -> Dict[int, float]:
        """
        Get the observed multiplier distribution.
        
        Returns:
            Dict mapping multiplier to observed probability
        """
        total = sum(self._multiplier_counts.values())
        if total == 0:
            return dict(DEFAULT_MULTIPLIER_PROBS)
        return {m: cnt / total for m, cnt in self._multiplier_counts.items()}

    def analyze_multiplier_deviation(self) -> dict:
        """
        Analyze if the observed multiplier distribution deviates from expected.
        
        Returns:
            dict with statistics and deviation analysis
        """
        observed = self.get_multiplier_distribution()
        total_obs = sum(self._multiplier_counts.values())
        
        # Calculate KL divergence
        kl_div = 0.0
        for m, expected in DEFAULT_MULTIPLIER_PROBS.items():
            obs = observed.get(m, 0.0)
            if obs > 0:
                kl_div += obs * math.log(obs / expected)
        
        # Calculate chi-squared statistic
        chi_sq = 0.0
        for m, expected in DEFAULT_MULTIPLIER_PROBS.items():
            obs_count = self._multiplier_counts.get(m, 0)
            expected_count = total_obs * expected
            if expected_count > 0:
                chi_sq += (obs_count - expected_count) ** 2 / expected_count
        
        # Determine significance
        deviation_level = "normal"
        if kl_div > 0.1:
            deviation_level = "mild"
        if kl_div > 0.3:
            deviation_level = "moderate"
        if kl_div > 0.5:
            deviation_level = "severe"
        
        return {
            "total_observations": total_obs,
            "kl_divergence": round(kl_div, 4),
            "chi_squared": round(chi_sq, 4),
            "deviation_level": deviation_level,
            "expected_distribution": {m: round(p, 4) for m, p in DEFAULT_MULTIPLIER_PROBS.items()},
            "observed_distribution": {m: round(p, 4) for m, p in observed.items()},
        }

    def analyze_landing_bias(self) -> dict:
        """
        Analyze if landing distribution shows significant bias from uniform.
        
        Returns:
            dict with bias analysis metrics
        """
        probs = self.get_landing_probs()
        uniform = 1.0 / NUM_SLOTS
        
        # Calculate TVD (Total Variation Distance)
        tvd = sum(abs(p - uniform) for p in probs) / 2
        
        # Calculate entropy
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
        max_entropy = math.log(NUM_SLOTS)
        entropy_ratio = entropy / max_entropy
        
        # Find hot/cold slots
        threshold = uniform * 1.5
        hot_slots = [i for i, p in enumerate(probs) if p >= threshold]
        cold_slots = [i for i, p in enumerate(probs) if p <= uniform * 0.5]
        
        bias_level = "none"
        if tvd > 0.05:
            bias_level = "mild"
        if tvd > 0.10:
            bias_level = "moderate"
        if tvd > 0.15:
            bias_level = "strong"
        
        return {
            "total_variation_distance": round(tvd, 4),
            "entropy_ratio": round(entropy_ratio, 4),
            "bias_level": bias_level,
            "hot_slots": [s + 1 for s in hot_slots],  # 1-indexed
            "cold_slots": [s + 1 for s in cold_slots],  # 1-indexed
            "slot_probabilities": [round(p, 4) for p in probs],
        }

    def get_machine_analysis(self) -> dict:
        """
        Get comprehensive machine analysis report.
        
        Returns:
            dict with all analysis metrics
        """
        mult_analysis = self.analyze_multiplier_deviation()
        bias_analysis = self.analyze_landing_bias()
        
        # Overall recommendation
        recommendations = []
        if mult_analysis["deviation_level"] == "severe":
            recommendations.append("倍率分布异常，建议换机器")
        if bias_analysis["bias_level"] == "strong":
            recommendations.append("检测到强落点偏差，可针对性投注")
        if self.total_plays < 30:
            recommendations.append("数据不足，建议继续记录")
        
        return {
            "total_plays": self.total_plays,
            "multiplier_analysis": mult_analysis,
            "landing_bias_analysis": bias_analysis,
            "recommendations": recommendations,
        }
