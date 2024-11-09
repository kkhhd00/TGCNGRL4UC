from pyomo.environ import *
import pyomo.environ as pyo

def ipopt_solve(load, a, b, mins, maxs, rampup, rampdown, power):
    model = ConcreteModel()
    num_gen = len(a)
    model.x = Var(range(num_gen), within=pyo.NonNegativeReals)
    model.obj = Objective(expr=sum(a[i] * model.x[i] ** 2 + b[i] * model.x[i] for i in range(num_gen)), sense=1)
    model.load_balance = Constraint(expr=sum(model.x[i] for i in range(num_gen)) == load)
    model.min_output = Constraint(range(num_gen), rule=lambda model, i: model.x[i] >= mins[i])
    model.max_output = Constraint(range(num_gen), rule=lambda model, i: model.x[i] <= maxs[i])
    def rampup_rule(model, i):
        if power[i] <= 0.1:
            return Constraint.Skip
        else:
            return model.x[i] - power[i] <= rampup[i]
    model.rampup_limit = Constraint(range(num_gen), rule=rampup_rule)
    model.rampdown_limit = Constraint(range(num_gen), rule=lambda model, i: power[i] - model.x[i] <= rampdown[i])
    solver = SolverFactory('ipopt')
    solver.solve(model)
    result = [model.x[i]() for i in range(num_gen)]
    return result
