import skfuzzy as fuzz
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class FuzzyVariable():
    """
    Implementation of variable fuzzification with input universe and thresholds.
    """
    def __init__(self, min:float, max:float, thresholds:list, functions:list, num_points:int):
        """
        Class initialisation.

        Parameters
        ----------
        min : float
            Minimum value of the variable universe
        max : float
            Maximum value of the variable universe
        thresholds : list of lists
            List with each element being the list of threshold of a domain, with length num_domains
        functions : list of callables
            List with each element being the membership function of a domain, with length num_domains
        num_points : int
            Number of points within variable universe.
        """
        self.thresholds = thresholds
        self.functions = functions
        self.num_points = num_points
        self.memberships = None
        self.num_domains = len(thresholds)
        self.universe = np.linspace(min, max, self.num_points)
        self.build()
    
    def check(self):
        """
        Checking parameters are correctly defined
        """
        assert len(self.functions) == self.num_domains

    def interpret(self, input:list, defuzz:str=None, plot:bool=False):

    def build(self):
        """
        Builds the membership functions based on input universe and thresholds.
        """
        self.memberships = []
        for t_domain in self.thresholds:
            self.memberships.append(self.choose_membership(t_domain)(self.universe, t_domain))

    def interpret_memberships(self, scalar):
        """
        Determine fuzzy relation matrix ``R`` using Mamdani implication for the
        fuzzy antecedent ``a`` and consequent ``b`` inputs.

        Parameters
        ----------
        scalar : float
            Scalar value within the universe.

        Returns
        -------
        membership_values : list
            List of membership values associated to the scalar for each membership function in order.

        """
        i=0
        while self.universe[i]<scalar:
            i+=1
        membership_values = []
        for membership in self.memberships:
            a, f_a = self.universe[i-2], membership[i-2]
            b, f_b = self.universe[i-1], membership[i-1]
            alpha = (f_a - f_b)/(a - b)
            beta = (a*f_b - b*f_a)/(a - b)
            membership_values.append(alpha*scalar+beta)
        return membership_values
    
    def choose_membership(self, thresholds:list):
        """
        Determine the membership function to use based on the users input.

        Parameters
        ----------
        thresholds : list
            List of thresholds to use for the membership function definition.

        Returns
        -------
        f : callable
            Membership function from scikit-fuzzy.

        """
        if len(thresholds) == 3:
            return fuzz.trimf
        elif len(thresholds) == 4:
            return fuzz.trapmf

            
        
class FuzzySystem():
    """
    Implementation of Mamdani fuzzy inference system.
    """
    def __init__(self, invar:list, outvar:FuzzyVariable, rules:np.array):
        """
        Class initialisation.

        Parameters
        ----------
        invar : list of FuzzyVariable
            Minimum value of the variable universe
        outvar : FuzzyVariable
            Maximum value of the variable universe
        rules : Nd array
            Association rule matrix of shape (invar1.num_domains, invar2.num_domains, ... , invarN.num_domains).
            Values type follow the condition that sorting would follow the same sorting than the outvar domains.
        """
        self.invar = invar
        self.outvar = outvar
        self.rules = rules
        self.out_domains = sorted(np.unique(self.rules))
        self.check()

    def check(self):
        """
        Checking rule definition follow the number of invar and outvar domains
        """
        assert len(self.out_domains) == self.outvar.num_domains
        for i, var in enumerate(self.invar):
            assert self.rules.shape[i] == var.num_domains

    def interpret(self, input:list, defuzz:str=None, plot:bool=False):
        """
        Determine output of fuzzy inference system from sclar valuers within invar universes.

        Parameters
        ----------
        input : list
            List of scalar values to put in invar. List length must be the length of invar.
        defuzz : str
            Mode of defuzzification function to use from scikit-fuzzy. 
            If None, output will be pre-defuzzification distribution.
        plot : bool
            If True, plotting pre-defuzzification distribution.

        Returns
        -------
        output : float / 1d array
            Output of fuzzy inference system if defuzz not None, or pre-defuzzification distribution if else.

        """
        memberships_in = []
        for i_var, value_var in enumerate(input):
            var = self.invar[i_var]
            memberships_var = var.interpret_memberships(value_var)
            memberships_in.append(np.array(memberships_var))
        memberships_in = np.array(memberships_in)
        
        # print(f"In memberships done.")
        memberships_out = []
        for i_memb_out, out_value in enumerate(self.out_domains):
            memberships_out_i = []
            positions = np.argwhere(self.rules == out_value)
            for position in positions:
                index_pos = tuple([[i for i in range(len(input))], position])
                memberships_out_i.append(np.min(memberships_in[index_pos]))
            memberships_out_i = np.max(memberships_out_i, axis=0)
            memberships_out.append(np.minimum(self.outvar.memberships[i_memb_out], memberships_out_i))
            # print(f"Out {i_memb_out} done.")

        # print(memberships_out)
        output = np.max(memberships_out, axis=0)
        if plot:
            plt.plot(self.outvar.universe, output)
            plt.ylim([0,1])
            plt.show()
        if not defuzz==None:
            return fuzz.defuzz(self.outvar.universe,
                               output,
                               mode=defuzz)
        
        return output

if __name__ == "__main__":
    num_points = 100
    energy_train = FuzzyVariable(min=0, max=1000, thresholds=[[0,0,25],[0,25,50],[25,50,1000,1000]], num_points=num_points)
    energy_test = FuzzyVariable(min=0, max=10, thresholds=[[0,0,2.5],[0,2.5,5],[2.5,5,10,10]], num_points=num_points)
    performance =  FuzzyVariable(min=0, max=1, thresholds=[[0,0,0.5],[0,0.5,1],[0.5,1,1]], num_points=num_points)
    score =  FuzzyVariable(min=0, max=1, thresholds=[[0,0,0.25],[0,0.25,0.5],[0.25,0.5,0.75],[0.5,0.75,1],[0.75,1,1]], num_points=num_points)

    rules = np.array([[[2,3,4],
                       [1,2,3],
                       [0,1,2]],
                      [[1,2,3],
                       [0,1,2],
                       [0,0,1]],
                      [[0,2,3],
                       [0,0,2],
                       [0,0,0]]])

    ScoreSys = FuzzySystem(invar=[energy_train,energy_test,performance], outvar=score, rules=rules)
    fuzz.interp_membership

    plot = False
    print(ScoreSys.interpret(input=[90,1,0.1], plot=plot, defuzz="centroid"))
    print(ScoreSys.interpret(input=[10,3,0.9], plot=plot, defuzz="centroid"))
    print(ScoreSys.interpret(input=[50,9,0.5], plot=plot, defuzz="centroid"))

