import numpy as np
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================
# Parameters for Monte Carlo Estimation
# ==============================================================

N = 600000      # 可调整：number of samples for area estimation
N_mc = 600000     # 可调整：number of samples for Monte Carlo PDF estimation

# Green parameter combinations（部分示例）
parameter_combinations = [
    (1, 1, np.pi/6, 1), (1, 1, np.pi/6, 2),
    (1, 1, np.pi/4, 1), (1, 1, np.pi/4, 2),
    (1, 1, np.pi/3, 1), (1, 1, np.pi/3, 2),
    (1, 2, np.pi/3, 1), (1, 2, np.pi/3, 2),
    (2, 1, np.pi/6, 1), (2, 1, np.pi/6, 2),
    (2, 1, np.pi/4, 1), (2, 1, np.pi/4, 2),
    (2, 1, np.pi/3, 1), (2, 1, np.pi/3, 2),
    (2, 2, np.pi/6, 1), (2, 2, np.pi/6, 2),
    (2, 2, np.pi/4, 1), (2, 2, np.pi/4, 2),
    (2, 2, np.pi/3, 1), (2, 2, np.pi/3, 2),
    (2, 3, np.pi/3, 1), (2, 3, np.pi/3, 2),
    (2, 4, np.pi/3, 1), (2, 4, np.pi/3, 2),
    (3, 1, np.pi/6, 1), (3, 1, np.pi/6, 2),
    (3, 1, np.pi/4, 1), (3, 1, np.pi/4, 2),
    (3, 1, np.pi/3, 1), (3, 1, np.pi/3, 2),
    (3, 2, np.pi/6, 1), (3, 2, np.pi/6, 2),
    (3, 2, np.pi/4, 1), (3, 2, np.pi/4, 2),
    (3, 2, np.pi/3, 1), (3, 2, np.pi/3, 2),
    (3, 3, np.pi/6, 1), (3, 3, np.pi/6, 2),
    (3, 3, np.pi/4, 1), (3, 3, np.pi/4, 2),
    (3, 3, np.pi/3, 1), (3, 3, np.pi/3, 2),
    (3, 4, np.pi/4, 1), (3, 4, np.pi/4, 2),
    (3, 4, np.pi/3, 1), (3, 4, np.pi/3, 2),
    (4, 1, np.pi/6, 1), (4, 1, np.pi/6, 2),
    (4, 1, np.pi/4, 1), (4, 1, np.pi/4, 2),
    (4, 1, np.pi/3, 1), (4, 1, np.pi/3, 2),
    (4, 2, np.pi/6, 1), (4, 2, np.pi/6, 2),
    (4, 2, np.pi/4, 1), (4, 2, np.pi/4, 2),
    (4, 2, np.pi/3, 1), (4, 2, np.pi/3, 2),
    (4, 3, np.pi/6, 1), (4, 3, np.pi/6, 2),
    (4, 3, np.pi/4, 1), (4, 3, np.pi/4, 2),
    (4, 3, np.pi/3, 1), (4, 3, np.pi/3, 2),
    (4, 4, np.pi/6, 1), (4, 4, np.pi/6, 2),
    (4, 4, np.pi/4, 1), (4, 4, np.pi/4, 2),
    (4, 4, np.pi/3, 1), (4, 4, np.pi/3, 2)
]

# ==============================================================
# Functions
# ==============================================================

def get_distributions(u1, u2, alpha, sigma):
    cov_matrix = [[sigma**2, 0], [0, sigma**2]]
    positive_center_1 = (u1 + u2 * np.cos(alpha), u2 * np.sin(alpha))
    positive_center_2 = (u1 - u2 * np.cos(alpha), -u2 * np.sin(alpha))
    negative_center_1 = (-u1 + u2 * np.cos(alpha), u2 * np.sin(alpha))
    negative_center_2 = (-u1 - u2 * np.cos(alpha), -u2 * np.sin(alpha))
    rv_positive1 = multivariate_normal(positive_center_1, cov_matrix)
    rv_positive2 = multivariate_normal(positive_center_2, cov_matrix)
    rv_negative1 = multivariate_normal(negative_center_1, cov_matrix)
    rv_negative2 = multivariate_normal(negative_center_2, cov_matrix)
    return rv_positive1, rv_positive2, rv_negative1, rv_negative2

def mixture_pdf(x, u1, u2, alpha, sigma):
    x1, x2 = x
    _, _, rv_n1, rv_n2 = get_distributions(u1, u2, alpha, sigma)
    return 0.5*rv_n1.pdf([x1, x2])+0.5*rv_n2.pdf([x1, x2])



def implicit_function(x, u1, u2, alpha, sigma):
    x1, x2 = x
    rv_p1, rv_p2, rv_n1, rv_n2 = get_distributions(u1, u2, alpha, sigma)
    f_pos = 0.5 * rv_p1.pdf([x1, x2]) + 0.5 * rv_p2.pdf([x1, x2])
    f_neg = 0.5 * rv_n1.pdf([x1, x2]) + 0.5 * rv_n2.pdf([x1, x2])
    return f_pos / f_neg - 1

def solve_x2_for_x1(x1, u1, u2, alpha, sigma):
    equation = lambda x2: implicit_function([x1, x2[0]], u1, u2, alpha, sigma)
    initial_guess = [0.1] if x1 >= 0 else [-0.1]
    x2_solution = fsolve(equation, initial_guess)
    return x2_solution[0]

def middle_line(x1, u1, u2, alpha):
    slope = (u1 - u2 * np.cos(alpha)) / (u2 * np.sin(alpha))
    return slope * x1

def monte_carlo_integration(vertical_widths, x1_min, x1_max):
    x1_range = x1_max - x1_min
    avg_height = np.mean(vertical_widths)
    return x1_range * avg_height

def estimate_avg_pdf(x1_min, x1_max, N_mc, u1, u2, alpha, sigma):
    x1_samples_mc = np.random.uniform(low=x1_min, high=x1_max, size=N_mc)
    x2_up_samples = middle_line(x1_samples_mc, u1, u2, alpha)
    x2_low_samples = np.array([solve_x2_for_x1(x1, u1, u2, alpha, sigma) for x1 in x1_samples_mc])
    r_mc = np.random.uniform(low=0, high=1, size=N_mc)
    x2_samples_mc = x2_low_samples + r_mc * (x2_up_samples - x2_low_samples)
    mis_region_points = np.column_stack((x1_samples_mc, x2_samples_mc))
    pdf_values_mc = np.array([mixture_pdf(point, u1, u2, alpha, sigma) for point in mis_region_points])
    return np.mean(pdf_values_mc)

# ==============================================================
# Main Loop: Output pdf_A_III and pdf_A_IV
# ==============================================================

print("u1\tu2\talpha\tsigma\tpdf_neg_III\tpdf_neg_IV")

for u1, u2, alpha, sigma in parameter_combinations:
    np.random.seed(42)

    # Right side region III
    x1_min = 0
    x1_max = u2 * np.cos(alpha)
    x2_min = 0

    x1_samples = np.random.uniform(low=x1_min, high=x1_max, size=N)
    x2_solutions = np.array([solve_x2_for_x1(x1, u1, u2, alpha, sigma) for x1 in x1_samples])
    vertical_widths = x2_solutions - x2_min
    estimated_area = monte_carlo_integration(vertical_widths, x1_min, x1_max)

    base = u2 * np.cos(alpha)
    height = middle_line(base, u1, u2, alpha)
    triangle_area = 0.5 * base * height
    mis_class_area = estimated_area - triangle_area

    # Region III
    pdf_B_III = estimate_avg_pdf(x1_min, x1_max, N_mc, u1, u2, alpha, sigma) * mis_class_area

    # Region IV
    x1_min_left = -u2 * np.cos(alpha)
    x1_max_left = 0
    pdf_B_IV = estimate_avg_pdf(x1_min_left, x1_max_left, N_mc, u1, u2, alpha, sigma) * mis_class_area

    relative_error = abs(pdf_B_III - pdf_B_IV) / (pdf_B_III + pdf_B_IV)
    print(f"{u1}\t{u2}\t{alpha:.2f}\t{sigma}\t{pdf_B_III:.8f}\t{pdf_B_IV:.8f}\t{relative_error:.8f}")


