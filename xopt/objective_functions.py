from abc import ABC, abstractmethod

class BaseObjectiveFunction(ABC):
    def __init__(self):
        self.variables = {}

    @abstractmethod
    def calculate(self, measured_values):
        pass

    # You can also have other common methods or attributes here
    
class default(BaseObjectiveFunction):
    def __init__(self,var=None):
        super().__init__()  # Call the parent constructor if needed
        self.variables = {}

    def calculate(self, measured_values):
        print(measured_values)
        value = list(measured_values['var1'].values())[0]
        return value

class ObjectiveFunction1(BaseObjectiveFunction):
    def __init__(self):
        super().__init__()  # Call the parent constructor if needed
        self.variables = {
                            "var1": {
                                "device_name": "UC_Amp2_IR_input",
                                "device_subscribe_variables": ["MeanCounts", "MaxCounts"]
                            },
                            "var2": {
                                "device_name": "UC_Amp3_IR_input",
                                "device_subscribe_variables": ["MeanCounts"]
                            }
                        }

    def calculate(self, measured_values):
        calculated_value=measured_values['var1']['MeanCounts']
        print('caculated value: ',calculated_value)
        return calculated_value
        
class HTUHexapodAlignmentSim(BaseObjectiveFunction):
    def __init__(self):
        super().__init__()  # Call the parent constructor if needed
        self.variables = {
                            "var1": {
                                "device_name": "UC_Amp2_IR_input",
                                "device_subscribe_variables": ["MeanCounts", "MaxCounts"]
                            },
                            "var2": {
                                "device_name": "UC_Amp3_IR_input",
                                "device_subscribe_variables": ["MeanCounts"]
                            }
                        }

    def calculate(self, measured_values):
        calculated_value=measured_values['var1']['MeanCounts']
        print('caculated value: ',calculated_value)
        return calculated_value



