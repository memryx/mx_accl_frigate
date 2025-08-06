import numpy as np

def get_port_info(dfp):
    """
    Returns
    input_info (dict) : a nested dictionary structured as follows
    {
        input_port_no_0 : {'raw_shape' : list,   # model input shape of fmap at port_no_0
                           'raw_dtype' : str     # dtype of fmap at port_no_0
                           'shape_shift_info': dict  { '+'            : list      # expand_dims at these indices
                                                       '-'            : list      # squeeze dims at these indices
                                                       'folded_optype'  : list     # ordered list of reshapes/transposes
                                                       'folded_opshape' : list of lists # ordered list of shapes for the corresponding folded_optytpe
                                                       }
                            'shape'     : list   # mxa input shape of fmap at port_no_0
                            }

        input_port_no_1 : ...
    }
    output_info (dict) : a nested dictionary structured as follows

    {
        output_port_no_0 :{'raw_shape' : list,   # model output shape of fmap at output_port_no_0
                           'raw_dtype' : str     # dtype of fmap at output_port_no_0
                           'shape_shift_info': dict  { '+'            : list      # expand_dims at these indices
                                                       '-'            : list      # squeeze dims at these indices
                                                       'folded_optype'  : list     # ordered list of reshapes/transposes
                                                       'folded_opshape' : list of lists # ordered list of shapes for the corresponding folded_optytpe
                                                       }
                            'shape'     : list   # mxa output shape of fmap at port_no_0
                          }


        output_port_no_1 : ...
    }
    """
    input_info = {}
    keys = ['raw_shape', 'raw_dtype', 'shape_shift_info', 'shape']
    for port_no, info in dfp.input_ports.items():
        input_info[port_no] = {}
        for k in keys:
            input_info[port_no][k] = info[k]

    output_info = {}
    for port_no, info in dfp.output_ports.items():
        output_info[port_no] = {}
        for k in keys:
            output_info[port_no][k] = info[k]

    return input_info, output_info

def convert_to_mxainput(input_info, x=None):
    """
    Used At input to convert from model_shape ---> mxa_shape

    Input:
    ------
    - input_info (dict) : a dictionary that contains a single port's input info
    
    input_port_no_0 :     {'raw_shape' : list,   # input shape of fmap at port_no_0
                           'raw_dtype' : str     # dtype of fmap at port_no_0
                           'shape_shift_info': dict  { '+'            : list      # expand_dims at these indices
                                                       '-'            : list      # squeeze dims at these indices
                                                       'folded_optype'  : list     # ordered list of reshapes/transposes
                                                       'folded_opshape' : list of lists # ordered list of shapes for the corresponding folded_optytpe}
                          'shape'     : list   # mxa input shape of fmap at port_no_0                              
                         }

    - x : input fmap of shape=model_shape. If None, create input of model_shape

    Order of Opertations!
    ---------------------
    1. input_info['shape_shift_info']['folded_optype'] : perform reshapes and transpose in order
    2. input_info['shape_shift_info']['+'] : expand_dims at indices present in the list
    3. input_info['shape_shift_info']['-'] : remove dims at indices present in the list

    Returns:
    x after Order of Operations are performed on x
    """
    model_shape = [s if isinstance(s,int) else 1 for s in input_info['raw_shape']]

    batch_dim = 0
    if len(input_info['shape_shift_info'].get('-', [])) > 0:
        batch_dim = 1

    # Check if x matches raw_shape
    if x is not None and list(x.shape[batch_dim:]) != model_shape[batch_dim:]:
        raise ValueError(f"Input shape {x.shape} provided does not match expected model's input shape {model_shape}.") from None
    
    if x is None:
        x = np.arange(np.prod(model_shape))
        x = np.reshape(x, model_shape)
    # reshape & transpose
    for op_type, op_shape in zip(input_info['shape_shift_info']['folded_optype'],input_info['shape_shift_info']['folded_opshape']):
        if op_type == 'reshape':
            x = np.reshape(x, op_shape)
        else:
            x = np.transpose(x, tuple(op_shape))
    # add dims
    x = np.expand_dims(x, axis=input_info['shape_shift_info'].get('+', []))

    # remove dims   
    x = np.squeeze(x, axis=tuple(input_info['shape_shift_info'].get('-', [])))

    return x

def convert_to_modeloutput(output_info, x=None):
    """
    Used at output to convert from mxa_shape ---> model_shape


    Input:
    ------
    output_info (dict) : a dictionary that contains a single port's output info

    output_port_no_0 :{'raw_shape' : list,   # output shape of fmap at output_port_no_0
                        'raw_dtype' : str     # dtype of fmap at output_port_no_0
                        'shape_shift_info': dict  { '+'            : list      # expand_dims at these indices
                                                    '-'            : list      # squeeze dims at these indices
                                                    'folded_optype'  : list     # ordered list of reshapes/transposes
                                                    'folded_opshape' : list of lists # ordered list of shapes for the corresponding folded_optytpe}
                        'shape'     : list   # mxa output shape of fmap at port_no_0                          
                       }



    - x : output fmap from mxa. If None, create fmap of mxa_outshape

    Order of Opertations!
    ---------------------
    1. output_info['shape_shift_info']['+'] : expand_dims at indices present in the list
    2. output_info['shape_shift_info']['-'] : remove dims at indices present in the list
    3. output_info['shape_shift_info']['folded_optype'] : perform reshapes and transpose in order

    Returns:
    x after Order of Operations are performed on x

    """
    mxa_shape = output_info['shape']
    if x is None:
        x = np.arange(np.prod(mxa_shape))
        x = np.reshape(x, mxa_shape)

    # add dims
    x = np.expand_dims(x, axis=output_info['shape_shift_info'].get('+', []))

    # remove dims
    x = np.squeeze(x, axis=tuple(output_info['shape_shift_info'].get('-', [])))

    # reshape & transpose
    for op_type, op_shape in zip(output_info['shape_shift_info']['folded_optype'],output_info['shape_shift_info']['folded_opshape']):
        if op_type == 'reshape':
            x = np.reshape(x, op_shape)
        else:
            x = np.transpose(x, tuple(op_shape))
    return x
