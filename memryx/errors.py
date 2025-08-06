from pprint import pformat

class OperatorError(Exception):
    def __init__(self, op_name=None, op_config=None, op_cond=None, message=None, identifier=None):
        self.message = message
        self.op_name = op_name
        self.op_cond = op_cond 

        if identifier is not None:
            layer_id = " ('" + identifier + "') "
        else:
            layer_id = ' '
        self.layer_id = layer_id

        if message is None:
            if op_name is not None and op_config is None and op_cond is None:
                message = f"{op_name}{layer_id}is not supported."
            elif op_name is not None and op_config is not None:
                fmt_config = pformat(op_config, indent=2).rstrip('\n')
                message = f"{op_name}{layer_id}with config: {fmt_config} is not supported"
            elif op_name is not None and op_cond is not None:
                message = f"{op_name}{layer_id}with {op_cond} is not supported."
            else:
                raise ValueError ('Invalid argument combination')
        elif message is not None and op_cond is None:
            if op_name is not None:
                message = f"{op_name}{layer_id}{message}"

        super().__init__(message)


class ResourceError(Exception):
    def __init__(self, resource=None, message="", increase_mpus=False, autocrop=False):

        if resource is not None:
            message = "Need more " + resource + ". " + message

        if increase_mpus:
            if autocrop is True:
                message = message + "Please try using more chips (-c , --num_chips)."
            else:
                message = message + "Please try autocrop (--autocrop) or using more chips (-c , --num_chips)."

        message += "\n"
        super().__init__(message)


class CompilerError(Exception):
    def __init__(self, stage=None, message=None, function=None):

        if message is None:
            message = "unspecified error"

        if stage is not None:
            if function is not None:
                message = stage + "::" + function + ": " + message
            else:
                message = stage + ": " + message
        elif function is not None:
            message = function + ": " + message

        message += "\n"
        super().__init__(message)


class SimulatorError(Exception):
    def __init__(self, message=None):

        if message is None:
            message = "unspecified error"

        message += "\n"
        super().__init__(message)


class MxaError(Exception):
    def __init__(self, message=None):

        if message is None:
            message = "unspecified error"

        message += "\n"
        super().__init__(message)


class UtilityError(Exception):
    def __init__(self, message=None):

        if message is None:
            message = "unspecified error"

        message += "\n"
        super().__init__(message)
