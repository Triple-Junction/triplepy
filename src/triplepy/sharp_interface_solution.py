import numpy as np
from scipy import optimize
import warnings


def calculate_slope(gb_energy_moving, gb_energy_stationary):
    return 1.0 / np.sqrt(4.0 * (gb_energy_moving / gb_energy_stationary)**2 - 1)


def calculate_dihedral_angle(gb_energy_moving, gb_energy_stationary):
    energy_ratio = gb_energy_stationary / gb_energy_moving
    return 2.0 * np.arccos(0.5*energy_ratio)


def calculate_gb_energy_ratio_from_dihedral_angle(theta):
    return 2.0 * np.cos(0.5 * theta)


def calculate_dihedral_angle_from_slope(m):
    return np.pi - 2.0*np.arctan(m)


def calculate_slope_from_dihedral_angle(theta):
    return np.tan(0.5*(np.pi - theta))


def isclose(a, b):
    return np.isclose(a, b)


def calc_l2_norm_rel_error(x_coarse, f_coarse, x_fine, f_fine):
    f_coarse_on_fine = np.interp(x_fine, x_coarse, f_coarse)
    diff2 = (f_coarse_on_fine - f_fine)*(f_coarse_on_fine - f_fine)
    return np.sqrt(np.trapz(diff2, x=x_fine) / np.trapz(np.array(f_fine)*np.array(f_fine), x=x_fine))


def integrate_trapezoidal(y0, x, slopes):
    ys = [y0]
    for i in range(len(x)-1):
        # trapezodial rule
        ys.append(ys[-1] + (slopes[i] + slopes[i+1]) * 0.5 * (x[i+1] - x[i]))
    return x, np.array(ys)


def integrate(y0, x, slopes):
    return integrate_trapezoidal(y0, x, slopes)


class GB_VelocityCalculator:
    def __init__(self, triple_junction_slope):
        self.m = triple_junction_slope
        self.sqrt_1_m2 = np.sqrt(1.0 + self.m * self.m)
        self.atan_m = np.arctan(self.m)

        # p, where v = 0
        self.p_stat = 2.0*self.m / self.sqrt_1_m2

        # p, where v = -p
        self.p_crit_neg1 = -2.0 * ((self.sqrt_1_m2 - 1.0)/self.m - self.atan_m)

        self.v_crit_neg1_m2 = np.log(1.0 + self.m * self.m) / self.m - 2.0 * self.atan_m
        # p, where v = -sqrt(1+m^2) p
        self.p_crit_neg1_m2 = -self.v_crit_neg1_m2 / self.sqrt_1_m2

        # we should have the following order p_crit_neg1_m2 < p_crit_neg1 < p_stat
        # p_crit_neg1 should always be lower than p_stat
        if self.p_crit_neg1 > self.p_stat:
            raise ValueError("The dimensional driving force for stationary GB (p_stat=%f) is larger than the one with q = -1 (p_neg1=%f)" % (self.p_stat, self.p_crit_neg1))

        # p_crit_neg1_m2 should always be lower than p_crit_neg1
        if self.p_crit_neg1_m2 > self.p_crit_neg1:
            raise ValueError("The dimensional driving force for q=-1 (p_neg1=%f) is larger than the one with q=-sqrt(1+m^2) (p_neg1_m2=%f)" % (self.p_crit_neg1, self.p_crit_neg1_m2))

        self.__run_tests()

    # "The most effective debugging techniques seem to be those which are designed and built into the program itself."
    # Donald E. Knuth
    # From Dr. Dobb's Journal, an article on debugging Windows programs, November 1993.
    def __run_tests(self):
        self.__test_velocity_neg1()
        self.__test_velocity_neg1_m2()

    def __test_velocity_neg1(self):
        if not isclose(self.calculate_velocity(self.p_crit_neg1), -self.p_crit_neg1):
            raise ValueError("The velocity for q=-1 (p_neg1=%f) is not identical to -p_neg1." % (self.p_crit_neg1))

    def __test_velocity_neg1_m2(self):
        if not isclose(self.calculate_velocity(self.p_crit_neg1_m2), self.v_crit_neg1_m2):
            raise ValueError("The velocity for q=-sqrt(1+m^2) (p_neg1_m2=%f) is not identical to -sqrt(1+m^2)*p_neg1_m2." % (self.p_crit_neg1_m2))

    def __calculate_velocity_lower_bound(self, p):
        # return lower bounds with decreasing order of p
        if p > self.p_stat:  # velocity is positive for p > p_stat
            return -1e-12
        if p > self.p_crit_neg1:  # velocity is larger -p_crit_neg1 for p_crit_neg1 < p < p_stat
            return -self.p_crit_neg1
        if p > self.p_crit_neg1_m2:  # velocity is larger v_crit_neg1_m2 for p_crit_neg1_m2 < p < p_crit_neg1
            return self.v_crit_neg1_m2
        if p > 0.0:  # velocity is larger -2 atan(m) for 0.0 < p < p_crit_neg1_m2
            return -2.0*self.atan_m
        # velocity is larger than this lower bound for negative p
        return np.minimum(-6.0*self.atan_m, -np.sqrt((1.0 + self.m*self.m)*p*p + 6.0*self.m*np.abs(p)))

    def __calculate_velocity_upper_bound(self, p):
        # return upper bounds with decreasing order of p
        if p > self.p_stat:  # velocity is smaller p for p > p_stat ( we add some correction since otherwise the root func is undefined )
            return p * (1.0 - 1.0e-12)
        if p > self.p_crit_neg1:  # velocity is below 0.0 for p_crit_neg1 < p < p_stat
            return 1e-12
        if p > self.p_crit_neg1_m2:  # velocity is below -p_crit_neg1 for p_crit_neg1_m2 < p < p_crit_neg1
            return -self.p_crit_neg1
        if p > 0.0:  # velocity is below v_crit_neg1_m2 for 0.0 < p < p_crit_neg1_m2
            return self.v_crit_neg1_m2
        # velocity is below -2.0*self.atan_m and below self.sqrt_1_m2*p for negative p
        return np.minimum(-2.0*self.atan_m, self.sqrt_1_m2*p*(1.0 - 1e-12))  # ( we add some correction since otherwise the root func is undefined )

    def __rootfunc(self, v, p):
        sqrt_vp = np.emath.sqrt(v*v - p*p)
        factor_1 = p / (v*sqrt_vp)
        # keep in mind that arctan(x/i)/i = -arctanh(x)
        return -np.real(factor_1 * (np.emath.arctanh(self.m*v / (self.sqrt_1_m2*sqrt_vp)) + np.emath.arctanh(self.m*p / sqrt_vp))) - self.atan_m/v - 0.5

    def __solve_single_velocity(self, p):
        if isclose(p, 0.0):
            return -2.0*self.atan_m

        if isclose(p, self.p_stat):
            return 0.0

        if isclose(p, self.p_crit_neg1):
            return -self.p_crit_neg1

        if isclose(p, self.p_crit_neg1_m2):
            return self.v_crit_neg1_m2

        lower_bound = self.__calculate_velocity_lower_bound(p)
        upper_bound = self.__calculate_velocity_upper_bound(p)
        lower_rootfunc = self.__rootfunc(lower_bound, p)
        upper_rootfunc = self.__rootfunc(upper_bound, p)

        # In case we have large driving forces, the true upper/lower bound cannot be used because of an undefined root function.
        # Since we correct with a relative factor of 1e-12 it may happen that the signs of the root function are not different.
        # This can only happen if the real velocity is above the estimated upper bound or below the estimated lower bound which means it has to be very near (< 1e-12 relative error) the linear regime.
        if np.sign(lower_rootfunc) == np.sign(upper_rootfunc):
            # Make the correction for large driving forces, leave unresolved otherwise which will raise an error in the brenth routine automatically.
            if p < 0.0:
                return self.sqrt_1_m2 * p
            elif p > self.p_stat:
                return p

        root = optimize.brenth(self.__rootfunc, lower_bound, upper_bound, args=(p))
        return root

    def calculate_velocity(self, p):
        if (isinstance(p, float) or isinstance(p, int)):
            return self.__solve_single_velocity(p)

        res = []
        for p_i in p:
            res.append(self.__solve_single_velocity(p_i))
        return np.array(res)


class GB_GeometrySolver:
    def __init__(self, triple_junction_slope, driving_force):
        calculator = GB_VelocityCalculator(triple_junction_slope)
        self.m = triple_junction_slope
        self.v = calculator.calculate_velocity(driving_force)
        self.p = driving_force

    def __xfunction(self, slope):
        v = self.v
        p = self.p
        # simple solution without driving force, exclude to allow dividing by p
        if isclose(p, 0.0):
            return np.arctan(slope) / v

        # exclude limiting special cases where denominator vanishes
        if isclose(v, 0.0):
            return -slope / (p * np.sqrt(1.0 + slope * slope))

        if isclose(v/p, -1.0):
            return np.where(isclose(slope, 0.0), 0.0, ((1.0 - np.sqrt(1.0 + slope * slope)) / slope + np.arctan(slope)) / v)

        sqrt_vp = np.emath.sqrt(v*v - p*p)  # can become imaginary
        result = np.full(slope.shape, np.nan)
        result[slope == self.m] = -0.5
        result[slope == -self.m] = 0.5
        mask = np.abs(slope) != self.m
        sqrt_slope = np.sqrt(1.0 + slope[mask]*slope[mask])
        # keep in mind that arctan(x/i)/i = -arctanh(x)
        result[mask] = np.real(p/(v*sqrt_vp) * (np.emath.arctanh((p*slope[mask])/sqrt_vp) + np.emath.arctanh((v*slope[mask])/(sqrt_vp * sqrt_slope)))) + np.arctan(slope[mask]) / v
        return result

    def __generate_slope_samples(self, n_samples):
        # generate slopes from 0 to -m
        slopes_intermediate = np.linspace(0.0, -0.9*self.m, n_samples)
        slopes_near_tj = np.concatenate((-self.m*(1.0-np.geomspace(0.1, 1e-9, n_samples))[1:], [-self.m]))
        return np.concatenate((slopes_intermediate, slopes_near_tj))

    def calc_dimensionless_geometry(self, relative_l2_tolerance=1.0e-8, max_refinements=100, kind="double"):
        # we start with only four samples in the beginning
        n_samples = 2
        slopes = self.__generate_slope_samples(n_samples)

        # calculate corresponding dimensionless x-coordinates
        xtilde = self.__xfunction(slopes)

        error = 0.0
        it = 0
        while ((it == 0 or error > relative_l2_tolerance) and it < max_refinements):
            it += 1

            x, y = integrate(0.0, xtilde, slopes)
            n_samples *= 2

            slopes_fine = self.__generate_slope_samples(n_samples)

            xtilde_fine = self.__xfunction(slopes_fine)

            x_fine, y_fine = integrate(0.0, xtilde_fine, slopes_fine)

            error = calc_l2_norm_rel_error(x, y, x_fine, y_fine)

            xtilde = xtilde_fine
            slopes = slopes_fine

        if error <= relative_l2_tolerance:
            print("Solution of GB geometry converged with L2 error = %.1e after %d refinements" % (error, it))
        else:
            warnings.warn("Solution did not converge! L2 error = %.1e" % error)

        # produce appropriate output for "kind"
        if kind == "half":
            return {"x": x_fine, "y": y_fine, "derivative": slopes_fine}
        if kind == "full":
            return {"x": np.concatenate((-np.flip(x_fine)[:-1], x_fine)),
                    "y": np.concatenate((np.flip(y_fine)[:-1], y_fine)),
                    "derivative": np.concatenate((-np.flip(slopes_fine)[:-1], slopes_fine))
                    }
        if kind == "double":
            # remove discontinuity of slope at triple junction
            slopes_fine[-1] = 0.0
            return {"x": np.concatenate((x_fine, 1.0 - np.flip(x_fine[:-1]))),
                    "y": np.concatenate((y_fine, np.flip(y_fine[:-1]))),
                    "derivative": np.concatenate((slopes_fine, -np.flip(slopes_fine[:-1])))
                    }
        else:
            raise ValueError("Unknown kind \"%s\"" % kind)
