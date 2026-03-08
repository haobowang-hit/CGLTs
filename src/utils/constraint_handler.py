"""
Multi-Objective Optimization - Constraint Handler Module
========================================================

This module implements constraint handling for multi-objective optimization
of CGLT designs, including geometric, manufacturing, and feasibility constraints.

Author: Enhanced for MOO based on CGLT-V1 framework
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
import warnings

class ConstraintHandler:
    """Handle geometric and manufacturing constraints for CGLT designs"""

    def __init__(self,
                 design_space_width: float = 120.0,
                 tolerance_margin: float = 2.0):
        """
        Initialize constraint handler

        Args:
            design_space_width: Fixed design space width D
            tolerance_margin: Minimum tolerance for manufacturing
        """
        self.design_space_width = design_space_width
        self.tolerance_margin = tolerance_margin

        # Parameter bounds (from paper)
        self.param_bounds = {
            'H': (24, 36),       # Height [24,36] mm, even numbers
            'Lumbus': (1, 10),   # Lumbus width [1,10] mm
            'Angle': (30, 80),   # Angle [30°,80°] with 5° increments
            'Radius': (10, 18),  # Radius [10,H/2] mm, but max 18 due to H≤36
        }

    def createside_constraint(self, H: float, Angle: float, Radius: float, Lumbus: float) -> Tuple[float, float]:
        """
        Geometric constraint validation function (createside from original code)

        Args:
            H: Height parameter
            Angle: Angle parameter in degrees
            Radius: Radius parameter
            Lumbus: Lumbus width parameter

        Returns:
            (r2, w1): Computed geometric parameters, (-1, -1) if invalid
        """
        try:
            sr = H / 2.0
            r1 = Radius
            a1 = Angle
            w = Lumbus

            # Geometric calculation
            cos_a1 = math.cos(math.radians(a1))
            denominator = 1 - cos_a1

            if abs(denominator) < 1e-10:
                return -1, -1

            r2 = (sr - r1 + r1 * cos_a1) / denominator

            # Design space constraint
            w1 = (self.design_space_width * 0.5 -
                  a1 * r1 * math.pi / 180 -
                  a1 * r2 * math.pi / 90 -
                  w * 0.5)

            return r2, w1

        except (ZeroDivisionError, ValueError, OverflowError):
            return -1, -1

    def validate_basic_bounds(self, design_params: np.ndarray) -> bool:
        """
        Validate basic parameter bounds

        Args:
            design_params: [8] array [H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2]

        Returns:
            True if all parameters within bounds
        """
        try:
            H1, Lumbus1, Angle1, Radius1 = design_params[:4]
            H2, Lumbus2, Angle2, Radius2 = design_params[4:]

            # Check bounds for both sections
            for H, Lumbus, Angle, Radius in [(H1, Lumbus1, Angle1, Radius1),
                                           (H2, Lumbus2, Angle2, Radius2)]:
                # H constraints: [24,36] mm, even numbers
                if not (24 <= H <= 36) or H % 2 != 0:
                    return False

                # Lumbus constraints: [1,10] mm
                if not (1 <= Lumbus <= 10):
                    return False

                # Angle constraints: [30°,80°]
                if not (30 <= Angle <= 80):
                    return False

                # Radius constraints: [10, H/2] mm
                if not (10 <= Radius <= H/2):
                    return False

            return True

        except (ValueError, IndexError):
            return False

    def validate_geometric_constraints(self, design_params: np.ndarray) -> bool:
        """
        Validate geometric constraints using createside function

        Args:
            design_params: [8] array [H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2]

        Returns:
            True if geometric constraints satisfied
        """
        try:
            H1, Lumbus1, Angle1, Radius1 = design_params[:4]
            H2, Lumbus2, Angle2, Radius2 = design_params[4:]

            # Check both sections
            for H, Angle, Radius, Lumbus in [(H1, Angle1, Radius1, Lumbus1),
                                            (H2, Angle2, Radius2, Lumbus2)]:
                r2, w1 = self.createside_constraint(H, Angle, Radius, Lumbus)

                # Geometric validity check
                if r2 <= self.tolerance_margin or w1 <= self.tolerance_margin:
                    return False

            return True

        except (ValueError, IndexError):
            return False

    def validate_manufacturing_constraints(self, design_params: np.ndarray) -> bool:
        """
        Validate manufacturing constraints

        Args:
            design_params: [8] array [H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2]

        Returns:
            True if manufacturing constraints satisfied
        """
        try:
            H1, Lumbus1, Angle1, Radius1 = design_params[:4]
            H2, Lumbus2, Angle2, Radius2 = design_params[4:]

            # Check manufacturability constraints
            for H, Lumbus, Angle, Radius in [(H1, Lumbus1, Angle1, Radius1),
                                           (H2, Lumbus2, Angle2, Radius2)]:
                # Minimum web thickness
                if Lumbus < self.tolerance_margin:
                    return False

                if Radius > H/2 - self.tolerance_margin:
                    return False

                # Angle manufacturability (avoid extreme values)
                if Angle < 32 or Angle > 78:
                    return False

            # Section transition constraints
            H_diff = abs(H1 - H2)
            if H_diff > 12:  # Avoid too abrupt transitions
                return False

            return True

        except (ValueError, IndexError):
            return False

    def validate_all_constraints(self, design_params: np.ndarray) -> bool:
        """
        Validate all constraints (basic bounds + geometric + manufacturing)

        Args:
            design_params: [8] array [H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2]

        Returns:
            True if all constraints satisfied
        """
        return (self.validate_basic_bounds(design_params) and
                self.validate_geometric_constraints(design_params) and
                self.validate_manufacturing_constraints(design_params))

    def compute_constraint_violations(self, design_params: np.ndarray) -> Dict[str, float]:
        """
        Compute constraint violations for penalty-based methods

        Args:
            design_params: [8] array [H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2]

        Returns:
            Dictionary with violation amounts (0 = feasible, >0 = violation)
        """
        violations = {}

        try:
            H1, Lumbus1, Angle1, Radius1 = design_params[:4]
            H2, Lumbus2, Angle2, Radius2 = design_params[4:]

            # Basic bound violations
            violations['H1_lower'] = max(0, 24 - H1)
            violations['H1_upper'] = max(0, H1 - 36)
            violations['H1_even'] = 0 if H1 % 2 == 0 else 1

            violations['H2_lower'] = max(0, 24 - H2)
            violations['H2_upper'] = max(0, H2 - 36)
            violations['H2_even'] = 0 if H2 % 2 == 0 else 1

            violations['Lumbus1_lower'] = max(0, 1 - Lumbus1)
            violations['Lumbus1_upper'] = max(0, Lumbus1 - 10)
            violations['Lumbus2_lower'] = max(0, 1 - Lumbus2)
            violations['Lumbus2_upper'] = max(0, Lumbus2 - 10)

            violations['Angle1_lower'] = max(0, 30 - Angle1)
            violations['Angle1_upper'] = max(0, Angle1 - 80)
            violations['Angle2_lower'] = max(0, 30 - Angle2)
            violations['Angle2_upper'] = max(0, Angle2 - 80)

            violations['Radius1_lower'] = max(0, 10 - Radius1)
            violations['Radius1_upper'] = max(0, Radius1 - H1/2)
            violations['Radius2_lower'] = max(0, 10 - Radius2)
            violations['Radius2_upper'] = max(0, Radius2 - H2/2)

            # Geometric constraint violations
            r2_1, w1_1 = self.createside_constraint(H1, Angle1, Radius1, Lumbus1)
            r2_2, w1_2 = self.createside_constraint(H2, Angle2, Radius2, Lumbus2)

            violations['r2_1_min'] = max(0, self.tolerance_margin - r2_1) if r2_1 > 0 else 1000
            violations['w1_1_min'] = max(0, self.tolerance_margin - w1_1) if w1_1 > 0 else 1000
            violations['r2_2_min'] = max(0, self.tolerance_margin - r2_2) if r2_2 > 0 else 1000
            violations['w1_2_min'] = max(0, self.tolerance_margin - w1_2) if w1_2 > 0 else 1000

            # Manufacturing constraint violations
            violations['mfg_Lumbus1'] = max(0, self.tolerance_margin - Lumbus1)
            violations['mfg_Lumbus2'] = max(0, self.tolerance_margin - Lumbus2)
            violations['mfg_Radius1'] = max(0, Radius1 - (H1/2 - self.tolerance_margin))
            violations['mfg_Radius2'] = max(0, Radius2 - (H2/2 - self.tolerance_margin))

            violations['mfg_Angle1_lower'] = max(0, 32 - Angle1)
            violations['mfg_Angle1_upper'] = max(0, Angle1 - 78)
            violations['mfg_Angle2_lower'] = max(0, 32 - Angle2)
            violations['mfg_Angle2_upper'] = max(0, Angle2 - 78)

            # Transition constraint
            violations['H_transition'] = max(0, abs(H1 - H2) - 12)

        except (ValueError, IndexError) as e:
            # If parameter extraction fails, assign large violations
            for key in ['basic', 'geometric', 'manufacturing']:
                violations[f'{key}_error'] = 1000

        return violations

    def compute_total_violation(self, design_params: np.ndarray) -> float:
        """
        Compute total constraint violation for penalty methods

        Args:
            design_params: [8] array with design parameters

        Returns:
            Total violation amount (0 = feasible)
        """
        violations = self.compute_constraint_violations(design_params)
        return sum(violations.values())

    def compute_feasibility_margin(self, design_params: np.ndarray) -> float:
        """
        Compute feasibility margin (distance to constraint boundary)

        Args:
            design_params: [8] array with design parameters

        Returns:
            Minimum margin to constraint boundary (larger = more robust)
        """
        if not self.validate_all_constraints(design_params):
            return 0.0

        try:
            H1, Lumbus1, Angle1, Radius1 = design_params[:4]
            H2, Lumbus2, Angle2, Radius2 = design_params[4:]

            margins = []

            # Basic bound margins
            margins.extend([
                H1 - 24, 36 - H1,
                H2 - 24, 36 - H2,
                Lumbus1 - 1, 10 - Lumbus1,
                Lumbus2 - 1, 10 - Lumbus2,
                Angle1 - 30, 80 - Angle1,
                Angle2 - 30, 80 - Angle2,
                Radius1 - 10, H1/2 - Radius1,
                Radius2 - 10, H2/2 - Radius2
            ])

            # Geometric margins
            r2_1, w1_1 = self.createside_constraint(H1, Angle1, Radius1, Lumbus1)
            r2_2, w1_2 = self.createside_constraint(H2, Angle2, Radius2, Lumbus2)

            if r2_1 > 0 and w1_1 > 0 and r2_2 > 0 and w1_2 > 0:
                margins.extend([
                    r2_1 - self.tolerance_margin,
                    w1_1 - self.tolerance_margin,
                    r2_2 - self.tolerance_margin,
                    w1_2 - self.tolerance_margin
                ])

            # Manufacturing margins
            margins.extend([
                Lumbus1 - self.tolerance_margin,
                Lumbus2 - self.tolerance_margin,
                (H1/2 - self.tolerance_margin) - Radius1,
                (H2/2 - self.tolerance_margin) - Radius2,
                Angle1 - 32, 78 - Angle1,
                Angle2 - 32, 78 - Angle2,
                12 - abs(H1 - H2)
            ])

            return min(margins)

        except (ValueError, IndexError):
            return 0.0

    def generate_feasible_sample(self,
                               n_samples: int = 1,
                               method: str = 'lhs',
                               max_attempts: int = 1000) -> np.ndarray:
        """
        Generate feasible design samples

        Args:
            n_samples: Number of samples to generate
            method: Sampling method ('lhs', 'random')
            max_attempts: Maximum attempts per sample

        Returns:
            Array of feasible design parameters [n_samples, 8]
        """
        feasible_samples = []

        for _ in range(n_samples):
            attempts = 0
            while attempts < max_attempts:
                # Generate candidate sample
                if method == 'lhs':
                    H1 = np.random.choice([24, 26, 28, 30, 32, 34, 36])
                    H2 = np.random.choice([24, 26, 28, 30, 32, 34, 36])
                    Lumbus1 = np.random.uniform(1, 10)
                    Lumbus2 = np.random.uniform(1, 10)
                    Angle1 = np.random.choice(range(30, 85, 5))
                    Angle2 = np.random.choice(range(30, 85, 5))
                    Radius1 = np.random.uniform(10, min(18, H1/2))
                    Radius2 = np.random.uniform(10, min(18, H2/2))
                else:  # random
                    H1 = np.random.choice([24, 26, 28, 30, 32, 34, 36])
                    H2 = np.random.choice([24, 26, 28, 30, 32, 34, 36])
                    Lumbus1 = np.random.uniform(1, 10)
                    Lumbus2 = np.random.uniform(1, 10)
                    Angle1 = np.random.uniform(30, 80)
                    Angle2 = np.random.uniform(30, 80)
                    Radius1 = np.random.uniform(10, min(18, H1/2))
                    Radius2 = np.random.uniform(10, min(18, H2/2))

                candidate = np.array([H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2])

                if self.validate_all_constraints(candidate):
                    feasible_samples.append(candidate)
                    break

                attempts += 1

            if attempts >= max_attempts:
                warnings.warn(f"Could not generate feasible sample after {max_attempts} attempts")
                # Use a default feasible design
                default_sample = np.array([30, 5, 45, 12, 30, 5, 45, 12])
                feasible_samples.append(default_sample)

        return np.array(feasible_samples)

    def repair_infeasible_design(self, design_params: np.ndarray) -> np.ndarray:
        """
        Repair an infeasible design to make it feasible

        Args:
            design_params: [8] array with potentially infeasible parameters

        Returns:
            Repaired feasible design parameters
        """
        repaired = design_params.copy()

        try:
            # Clip to basic bounds
            repaired[0] = np.clip(repaired[0], 24, 36)  # H1
            repaired[4] = np.clip(repaired[4], 24, 36)  # H2

            # Ensure even H values
            repaired[0] = 2 * round(repaired[0] / 2)  # H1
            repaired[4] = 2 * round(repaired[4] / 2)  # H2

            repaired[1] = np.clip(repaired[1], 1, 10)   # Lumbus1
            repaired[5] = np.clip(repaired[5], 1, 10)   # Lumbus2

            repaired[2] = np.clip(repaired[2], 30, 80)  # Angle1
            repaired[6] = np.clip(repaired[6], 30, 80)  # Angle2

            # Radius constraints depend on H
            repaired[3] = np.clip(repaired[3], 10, min(18, repaired[0]/2))  # Radius1
            repaired[7] = np.clip(repaired[7], 10, min(18, repaired[4]/2))  # Radius2

            # Verify and further adjust if needed
            if not self.validate_all_constraints(repaired):
                # Use conservative values
                H1, H2 = repaired[0], repaired[4]
                repaired = np.array([
                    H1, 5.0, 45.0, min(12.0, H1/2 - 2),
                    H2, 5.0, 45.0, min(12.0, H2/2 - 2)
                ])

        except Exception:
            # Fallback to default feasible design
            repaired = np.array([30, 5, 45, 12, 30, 5, 45, 12])

        return repaired


class ConstraintAwareSampler:
    """Constraint-aware sampling for optimization algorithms"""

    def __init__(self, constraint_handler: ConstraintHandler):
        """
        Initialize constraint-aware sampler

        Args:
            constraint_handler: ConstraintHandler instance
        """
        self.constraint_handler = constraint_handler

    def sample_within_constraints(self,
                                n_samples: int,
                                base_distribution: str = 'uniform',
                                repair_method: str = 'clip') -> np.ndarray:
        """
        Sample designs within constraints

        Args:
            n_samples: Number of samples
            base_distribution: Base sampling distribution
            repair_method: Method for handling infeasible samples

        Returns:
            Array of feasible samples [n_samples, 8]
        """
        if base_distribution == 'uniform':
            samples = self.constraint_handler.generate_feasible_sample(
                n_samples=n_samples, method='random'
            )
        elif base_distribution == 'lhs':
            samples = self.constraint_handler.generate_feasible_sample(
                n_samples=n_samples, method='lhs'
            )
        else:
            raise ValueError(f"Unknown distribution: {base_distribution}")

        return samples

    def evaluate_with_penalties(self,
                              design_params: np.ndarray,
                              objective_values: np.ndarray,
                              penalty_weight: float = 1000.0) -> np.ndarray:
        """
        Add constraint violations as penalties to objectives

        Args:
            design_params: Design parameters
            objective_values: Original objective values
            penalty_weight: Weight for constraint violations

        Returns:
            Penalized objective values
        """
        total_violation = self.constraint_handler.compute_total_violation(design_params)
        penalty = penalty_weight * total_violation

        # Add penalty to all objectives (assuming minimization)
        penalized = objective_values + penalty

        return penalized