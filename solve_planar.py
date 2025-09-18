import sympy as sp


class SolvePlanarSystem:
    def __init__(self, num_forces: int, case: int):
        self.num_forces = num_forces
        self.case = case


        # Define symbols
        self.F_x, self.F_y = sp.symbols('F_x F_y')
        self.F_x2, self.F_y2 = sp.symbols('F_x2 F_y2')
        self.l, self.q1 = sp.symbols('l q1')


        # Known totals (inputs)
        self.F_tot_x, self.F_tot_y = sp.symbols('F_tot_x F_tot_y')
        self.tau_1, self.tau_2, self.tau_3, self.tau_4 = sp.symbols('tau_1 tau_2 tau_3 tau_4')


        # Storage
        self.eqns = []
        self.vars = []


        # Build system
        self._build_system()


    def _build_system(self):
        if self.num_forces == 1:
            if self.case == 1:
                print('entered case 1')
                eq1 = sp.Eq(self.F_tot_x, self.F_x)
                eq2 = sp.Eq(self.F_tot_y, self.F_y)
                eq3 = sp.Eq(self.tau_1, self.l * (-sp.sin(self.q1) * self.F_x + sp.cos(self.q1) * self.F_y))
                eq4 = sp.Eq(self.tau_2, 0)
                eq5 = sp.Eq(self.tau_3, 0)
                eq6 = sp.Eq(self.tau_4, 0)
                self.vars = [self.F_x, self.F_y, self.l]


            elif self.case == 2:
                eq1 = sp.Eq(self.F_tot_x, self.F_x2)
                eq2 = sp.Eq(self.F_tot_y, self.F_y2)
                eq3 = sp.Eq(self.tau_1, self.l * (-sp.sin(self.q1) * self.F_x + sp.cos(self.q1) * self.F_y))
                eq4 = sp.Eq(self.tau_2, 0)
                eq5 = sp.Eq(self.tau_3, 0)
                eq6 = sp.Eq(self.tau_4, 0)
                self.vars = [self.F_x2, self.F_y2, self.l]


    # Cases 3 and 4 can be added similarly


    # elif self.num_forces == 2:
    # if self.case in [14, 24, 34, 44]:
    # Placeholder for two-force cases
        # eq1 = eq2 = eq3 = eq4 = eq5 = eq6 = None
        # self.vars = []


        self.eqns = [eq1, eq2, eq3, eq4, eq5, eq6]


    def solve(self, knowns: dict):
        """Solve system given dict of known values for totals and parameters."""
        eqns_sub = [eq.subs(knowns) for eq in self.eqns if eq is not None]
        print('eqns_sub: ', eqns_sub)
        sol = sp.solve(eqns_sub, self.vars, dict=True)
        return sol
        
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