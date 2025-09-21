import sympy as sp
import numpy as np
import time


class SolvePlanarSystem:
    def __init__(self, num_forces: int, case: int):
        self.num_forces = num_forces
        self.case = case
        print("num_forces:", self.num_forces, "case:", self.case)



        # Define symbols
        self.F_x, self.F_y, self.l_bar = sp.symbols('F_x F_y l') # unkowns: external single force and contact length
        self.Fa_x, self.Fa_y, self.la_bar = sp.symbols('Fa_x Fa_y la_bar') # contact force a
        self.Fb_x, self.Fb_y, self.lb_bar = sp.symbols('Fb_x Fb_y lb_bar') # contact force b

        self.q1, self.q2, self.q3, self.q4 = sp.symbols('q1 q2 q3 q4') # robot state
        self.l1, self.l2, self.l3, self.l4 = sp.symbols('l1 l2 l3 l4') # link lengths


        # Known totals (inputs)
        self.F_tot_x, self.F_tot_y = sp.symbols('F_tot_x F_tot_y')
        self.tau_1, self.tau_2, self.tau_3, self.tau_4 = sp.symbols('tau_1 tau_2 tau_3 tau_4')


        # Storage
        self.eqns = []
        self.vars = []


        # Build system
        #self._build_system()


    def _build_system(self):
        if self.num_forces == 1:
            if self.case == 1:
                #print('entered case 1')
                eq1 = sp.Eq(self.F_tot_x, self.F_x)
                eq2 = sp.Eq(self.F_tot_y, self.F_y)
                eq3 = sp.Eq(self.tau_1, self.l_bar * (-sp.sin(self.q1) * self.F_x + sp.cos(self.q1) * self.F_y))
                eq4 = sp.Eq(self.tau_2, 0)
                eq5 = sp.Eq(self.tau_3, 0)
                eq6 = sp.Eq(self.tau_4, 0)
                self.eqns = [eq1, eq2, eq3, eq4, eq5, eq6]
                self.vars = [self.F_x, self.F_y, self.l_bar]


            elif self.case == 2:
                #print('entered case 2')
                eq1 = sp.Eq(self.F_tot_x, self.F_x)
                eq2 = sp.Eq(self.F_tot_y, self.F_y)
                eq3 = sp.Eq(self.tau_1, self.l1 * (-sp.sin(self.q1) * self.F_x + sp.cos(self.q1) * self.F_y)+self.tau_2)
                eq4 = sp.Eq(self.tau_2, self.l_bar*(-sp.sin(self.q1+self.q2)*self.F_x + sp.cos(self.q1+self.q2)*self.F_y))
                eq5 = sp.Eq(self.tau_3, 0)
                eq6 = sp.Eq(self.tau_4, 0)
                self.eqns = [eq1, eq2, eq4, eq5, eq6]
                self.vars = [self.F_x, self.F_y, self.l_bar]

            elif self.case == 3:
                #print('entered case 3')
                eq1 = sp.Eq(self.F_tot_x, self.F_x)
                eq2 = sp.Eq(self.F_tot_y, self.F_y)
                eq3 = sp.Eq(self.tau_1, self.l1 * (-sp.sin(self.q1) * self.F_x + sp.cos(self.q1) * self.F_y)+self.tau_2)
                eq4 = sp.Eq(self.tau_2, self.l2 * (-sp.sin(self.q1+self.q2) * self.F_x + sp.cos(self.q1+self.q2) * self.F_y)+self.tau_3)
                eq5 = sp.Eq(self.tau_3, self.l_bar * (-sp.sin(self.q1+self.q2+self.q3) * self.F_x + sp.cos(self.q1+self.q2+self.q3) * self.F_y))
                eq6 = sp.Eq(self.tau_4, 0)
                self.eqns = [eq1, eq2, eq5, eq6]
                self.vars = [self.F_x, self.F_y, self.l_bar]

            elif self.case == 4:
                #print('entered case 4')
                eq1 = sp.Eq(self.F_tot_x, self.F_x)
                eq2 = sp.Eq(self.F_tot_y, self.F_y)
                eq3 = sp.Eq(self.tau_1, self.l1 * (-sp.sin(self.q1) * self.F_x + sp.cos(self.q1) * self.F_y)+self.tau_2)
                eq4 = sp.Eq(self.tau_2, self.l2 * (-sp.sin(self.q1+self.q2) * self.F_x + sp.cos(self.q1+self.q2) * self.F_y)+self.tau_3)
                eq5 = sp.Eq(self.tau_3, self.l3 * (-sp.sin(self.q1+self.q2+self.q3) * self.F_x + sp.cos(self.q1+self.q2+self.q3) * self.F_y)+self.tau_4)
                eq6 = sp.Eq(self.tau_4, self.l_bar * (-sp.sin(self.q1+self.q2+self.q3+self.q4) * self.F_x + sp.cos(self.q1+self.q2+self.q3+self.q4) * self.F_y))
                self.eqns = [eq1, eq2, eq6]
                self.vars = [self.F_x, self.F_y, self.l_bar]


        # elif self.num_forces == 2:
        #     if self.case == 14:
        #         print('entered case 14')
        #         eq1 = sp.Eq(self.F_tot_x, self.Fa_x + self.Fb_x)
        #         eq2 = sp.Eq(self.F_tot_y, self.Fa_y + self.Fb_y)
        #         eq3 = sp.Eq(self.tau_1, self.l1 * (-sp.sin(self.q1) * self.Fb_x + sp.cos(self.q1) * self.Fb_y)+self.la_bar * (-sp.sin(self.q1) * self.Fa_x + sp.cos(self.q1) * self.Fa_y) + self.tau_2)
        #         eq4 = sp.Eq(self.tau_2, self.l2 * (-sp.sin(self.q1+self.q2) * self.Fb_x + sp.cos(self.q1+self.q2) * self.Fb_y) + self.tau_3)
        #         eq5 = sp.Eq(self.tau_3, self.l3 * (-sp.sin(self.q1+self.q2+self.q3) * self.Fb_x + sp.cos(self.q1+self.q2+self.q3) * self.Fb_y) + self.tau_3)
        #         eq6 = sp.Eq(self.tau_4, self.lb_bar * (-sp.sin(self.q1+self.q2+self.q3+self.q4) * self.Fb_x + sp.cos(self.q1+self.q2+self.q3+self.q4) * self.Fb_y))
        #         self.eqns = [eq4, eq5]
        #         self.vars = [self.Fb_x, self.Fb_y]
        #         sol = sp.solve(self.eqns, self.vars, dict=True)
        #         self.Fb_x = sol[self.Fb_x][0]
        #         self.Fb_y = sol[self.Fb_y][0]
        #         self.eqns = [eq1, eq2, eq3, eq6]
        #         self.vars = [self.Fa_x, self.Fa_y, self.la_bar, self.lb_bar]
        #         sol = sp.solve(self.eqns, self.vars, dict=True)

        elif self.num_forces == 2:
            if self.case == 14:
                print('entered case 14')

                # Define equations
                eq1 = sp.Eq(self.F_tot_x, self.Fa_x + self.Fb_x)
                eq2 = sp.Eq(self.F_tot_y, self.Fa_y + self.Fb_y)
                eq3 = sp.Eq(
                    self.tau_1,
                    self.l1 * (-sp.sin(self.q1) * self.Fb_x + sp.cos(self.q1) * self.Fb_y)
                    + self.la_bar * (-sp.sin(self.q1) * self.Fa_x + sp.cos(self.q1) * self.Fa_y)
                    + self.tau_2
                )
                eq4 = sp.Eq(
                    self.tau_2,
                    self.l2 * (-sp.sin(self.q1 + self.q2) * self.Fb_x + sp.cos(self.q1 + self.q2) * self.Fb_y)
                    + self.tau_3
                )
                eq5 = sp.Eq(
                    self.tau_3,
                    self.l3 * (-sp.sin(self.q1 + self.q2 + self.q3) * self.Fb_x + sp.cos(self.q1 + self.q2 + self.q3) * self.Fb_y)
                    + self.tau_4
                )
                eq6 = sp.Eq(
                    self.tau_4,
                    self.lb_bar * (-sp.sin(self.q1 + self.q2 + self.q3 + self.q4) * self.Fb_x + sp.cos(self.q1 + self.q2 + self.q3 + self.q4) * self.Fb_y)
                )

                self.eqns = [eq1, eq2, eq3, eq4, eq5, eq6]
                self.vars = [self.Fa_x, self.Fa_y, self.Fb_x, self.Fb_y, self.la_bar, self.lb_bar]
      
                # Step 1: solve eq4 and eq5 for Fb_x, Fb_y
                # time_0 = time.time()
                # sol_fb = sp.solve([eq4, eq5], [self.Fb_x, self.Fb_y], dict=True)
                # time_1 = time.time()
                # print(f'Time to solve for Fb_x, Fb_y: {time_1 - time_0}')

                # if sol_fb:
                #     sol_fb = sol_fb[0]  # take first solution dict

                #     # Step 2: substitute Fb_x, Fb_y into the remaining equations
                #     eqs_remaining = [eq1, eq2, eq3, eq6]
                #     eqs_remaining = [eq.subs(sol_fb) for eq in eqs_remaining]

                #     # Step 3: solve for Fa_x, Fa_y, la_bar, lb_bar
                #     time_0 = time.time()
                #     sol_rest = sp.solve(eqs_remaining, [self.Fa_x, self.Fa_y, self.la_bar, self.lb_bar], dict=True)
                #     time_1 = time.time()
                #     print(f'Time to solve for Fa_x, Fa_y, la_bar, lb_bar: {time_1 - time_0}')

                #     # Full solution is a combination of both
                #     self.solution = []
                #     for s in sol_rest:
                #         combined = {**s, **sol_fb}
                #         self.solution.append(combined)
                # else:
                #     self.solution = []


                # --- STEP A: solve eq4 & eq5 for Fb_x, Fb_y (linear system) ---
                # time_0 = time.time()
                # sol_fb_list = sp.solve([eq4, eq5], [self.Fb_x, self.Fb_y], dict=True, simplify=True)
                # time_1 = time.time()
                # print(f"Time to solve eq4 & eq5: {time_1 - time_0}")

                # if not sol_fb_list:
                #     # fallback: use linear_eq_to_matrix to detect singularity and solve
                #     A, b = sp.linear_eq_to_matrix([eq4, eq5], [self.Fb_x, self.Fb_y])
                #     A_det = sp.simplify(A.det())
                #     if A_det == 0:
                #         # degenerate case: equations are dependent -> infinite or no solutions
                #         print("Warning: torque equations are singular (dependant). A.det == 0")
                #         self.solution = []   # or handle parameterised solution with sp.solve([...], ...)
                #         return
                #     sol_vec = A.LUsolve(b)   # gives vector solution
                #     sol_fb = {self.Fb_x: sp.simplify(sol_vec[0]), self.Fb_y: sp.simplify(sol_vec[1])}
                # else:
                #     sol_fb = sol_fb_list[0]  # take first solution (often the only one)

                # # --- STEP B: Fa_x, Fa_y from eq1 & eq2 (trivial linear) ---
                # Fa_x_expr = sp.simplify(self.F_tot_x - sol_fb[self.Fb_x])
                # Fa_y_expr = sp.simplify(self.F_tot_y - sol_fb[self.Fb_y])

                # # --- STEP C: lb_bar from eq6 (linear in lb_bar) ---
                # sol_lb_list = sp.solve(eq6.subs(sol_fb), [self.lb_bar], dict=True, simplify=True)
                # if sol_lb_list:
                #     sol_lb = sol_lb_list[0]
                # else:
                #     # eq6 might be 0 * lb_bar = tau4 (degenerate) -> check multiplier
                #     mult_expr = sp.simplify(-sp.sin(self.q1 + self.q2 + self.q3 + self.q4) * sol_fb[self.Fb_x] +
                #                             sp.cos(self.q1 + self.q2 + self.q3 + self.q4) * sol_fb[self.Fb_y])
                #     if mult_expr == 0:
                #         print("Warning: lb_bar multiplier is zero -> lb_bar undetermined or inconsistent.")
                #         sol_lb = {}
                #     else:
                #         sol_lb = {}

                # # --- STEP D: la_bar from eq3 (substitute Fa expressions and Fb) ---
                # subs_for_eq3 = {
                #     self.Fb_x: sol_fb[self.Fb_x],
                #     self.Fb_y: sol_fb[self.Fb_y],
                #     self.Fa_x: Fa_x_expr,
                #     self.Fa_y: Fa_y_expr
                # }
                # sol_la_list = sp.solve(eq3.subs(subs_for_eq3), [self.la_bar], dict=True, simplify=True)
                # sol_la = sol_la_list[0] if sol_la_list else {}

                # # --- ASSEMBLE final solution dict ---
                # full_sol = {
                #     self.Fb_x: sol_fb[self.Fb_x],
                #     self.Fb_y: sol_fb[self.Fb_y],
                #     self.Fa_x: Fa_x_expr,
                #     self.Fa_y: Fa_y_expr
                # }
                # full_sol.update(sol_lb)
                # full_sol.update(sol_la)

                # self.solution = [full_sol] if full_sol else []
                # print("Solution (piecewise):", self.solution)


    # Cases 3 and 4 can be added similarly


    # elif self.num_forces == 2:
    # if self.case in [14, 24, 34, 44]:
    # Placeholder for two-force cases
        # eq1 = eq2 = eq3 = eq4 = eq5 = eq6 = None
        # self.vars = []


            #self.eqns = [eq1, eq2, eq3, eq4, eq5, eq6]



    # solver for SymPy expressions
    def solve(self, knowns: dict):
        """Solve system given dict of known values for totals and parameters."""
        self._build_system()
        eqns_sub = [eq.subs(knowns) for eq in self.eqns if eq is not None]
        #print('eqns_sub: ', eqns_sub)
        print('vars: ', self.vars)
        sol = sp.solve(eqns_sub, self.vars, dict=True)
        return sol
    
    def block_solve(equations, variables, knowns: dict):
        equations = [sp.Eq(eq.lhs - eq.rhs, 0) for eq in equations]  # normalize
        equations = [eq.subs(knowns) for eq in equations if eq is not None]

        lin_eqs = []
        other_eqs = []
        lin_vars = set()
        
        # Split linear vs nonlinear equations
        for eq in equations:
            poly = sp.Poly(eq.lhs, *variables, domain='EX')
            if poly.is_linear:
                lin_eqs.append(eq)
                lin_vars |= set(poly.gens)
            else:
                other_eqs.append(eq)
        
        final_solution = {}
        
        if lin_eqs:
            # Convert to matrix form
            A, b = sp.linear_eq_to_matrix(lin_eqs, list(lin_vars))
            A_np = np.array(A, dtype=float)
            b_np = np.array(b, dtype=float).reshape(-1)
            
            # Solve (works for square, over/underdetermined)
            sol_lin, *_ = np.linalg.lstsq(A_np, b_np, rcond=None)
            final_solution.update({var: val for var, val in zip(lin_vars, sol_lin)})
        
        # Now solve remaining nonlinear equations
        for eq in other_eqs:
            eq_sub = eq.subs(final_solution)
            unknowns = [v for v in eq_sub.free_symbols if v not in final_solution]
            if unknowns:
                sol = sp.solve(eq_sub, unknowns)
                # Pick first solution (extendable)
                if sol:
                    if isinstance(sol, dict):
                        for k, v in sol.items():
                            final_solution[k] = float(v)
                    else:
                        final_solution[unknowns[0]] = float(sol[0])
        
        return final_solution

        
        # Initialize for num_forces = 1, case = 1
solver = SolvePlanarSystem(num_forces=1, case=1)

# Provide known values (example numbers)
knowns = {
    solver.F_tot_x: 10,     # known total force x
    solver.F_tot_y: 5,      # known total force y
    solver.tau_1: 2,        # known torque
    solver.q1: sp.pi/4      # orientation
}

# Solve system
solution = solver.solve(knowns)

print(solution)


solver_2_forces = SolvePlanarSystem(num_forces=2, case=14)

knowns = {
 solver_2_forces.F_tot_x: 0,
 solver_2_forces.F_tot_y: 0,
 solver_2_forces.tau_1: 13,
 solver_2_forces.tau_2: 0,
 solver_2_forces.tau_3: 12,
 solver_2_forces.tau_4: 0,
 solver_2_forces.q1: np.pi,
 solver_2_forces.q2: 0,
 solver_2_forces.q3: 3,
 solver_2_forces.q4: 0,
 solver_2_forces.l1: 1,
 solver_2_forces.l2: 1,
 solver_2_forces.l3: 1,
 solver_2_forces.l4: 1
}

print('eqns: ', solver_2_forces.eqns)
print('vars: ', solver_2_forces.vars)
print('knowns: ', knowns)
sol = solver_2_forces.block_solve(solver_2_forces.eqns, solver_2_forces.vars, knowns)

print(sol)