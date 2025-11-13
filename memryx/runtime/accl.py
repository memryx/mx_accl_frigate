import inspect
from dataclasses import dataclass, astuple
import io
import logging
import numbers
import os
import queue
import sys
import threading
import time
import traceback
import weakref
from abc import ABC
from collections import defaultdict, deque
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np

try:
    import memryx
except ImportError:
    mix_home = Path(__file__).parent.parent.resolve()
    sys.path.append(str(mix_home))
    import memryx

from memryx.errors import MxaError
from memryx.runtime.accl_pre_post_model_loader import ModelLoader
from memryx.runtime.dfp import Dfp
from memryx.runtime.dfp_runner import DFPRunner
from memryx.utilities.dfp_shape_convert import (
    convert_to_modeloutput,
    convert_to_mxainput,
    get_port_info,
)

logger = logging.getLogger(__name__)
handler = logging.NullHandler()
formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(threadName)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
handler.setFormatter(formatter)
logger.addHandler(handler)

CHIP_NAME_TO_GEN = {"Cascade": 3, "Cascade+": 3.1}


def advance_iter(iterator):
    try:
        res = next(iterator)
    except StopIteration:
        res = None
    return res

def validate_input_callback_result(input_frames):
    if input_frames is None:
        return
    if not isinstance(input_frames, (list, tuple)):
        frame_seq = [input_frames]
    else:
        frame_seq = input_frames
    for x in frame_seq:
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"input_callback must return a `np.ndarray` or sequence of `np.ndarray`, got: {type(x)}"
            )

class _PassThroughModel:
    def __init__(self):
        pass

    def predict_for_mxa(
        self, inputs, use_model_shape, pre_model=False, output_order=None
    ):
        """return input as is, kwargs only to match interface"""
        _, _, _ = use_model_shape, pre_model, output_order
        if isinstance(inputs, dict):
            return list(inputs.values())
        return inputs


@dataclass
class SchedulerOptions:
    """
    Configuration options for the scheduler.

    Parameters
    ----------
    frame_limit : int, optional
        Number of frames to process before swapping out. Default is 20.

    time_limit : int, optional
        Maximum idle time (in milliseconds) before swapping out. Default is 250.

    stop_on_empty : bool, optional
        **REMOVED IN SDK 2.1** Always is False.

    ifmap_queue_size : int, optional
        Size of the shared input feature map queue. Default is 16.

    ofmap_queue_size : int, optional
        Size of the per-client output feature map queues. Default is 21.

    """
    frame_limit: int = 20
    time_limit: int = 250
    stop_on_empty: bool = False
    ifmap_queue_size: int = 16
    ofmap_queue_size: int = 21


@dataclass
class ClientOptions:
    """
    Client runtime behavior configuration.

    Parameters
    ----------
    smoothing : bool, optional
        Whether to enable FPS smoothing. Default is False.

    fps_target : float, optional
        Target frames per second. A delay of 1 / fps_target seconds is enforced between frames. Default is 0.0.

    """
    smoothing: bool = False
    fps_target: float = 0.0


class Accl(ABC):
    def __init__(
        self,
        dfp: Union[str, Path, bytes, Dfp],
        device_ids: Union[List[int], int]=[0],
        use_model_shape: Tuple[bool, bool]=(True, True),
        local_mode: bool = False,
        scheduler_options: SchedulerOptions =  SchedulerOptions(),
        client_options: ClientOptions = ClientOptions(),
        manager_addr: str = "/run/mxa_manager/",
        manager_port: int = 10000,
        ignore_manager: bool = False,
    ):
        logger.info(f"Run in {'local' if local_mode else 'shared'} mode")


        ##### Import python bindings for mxa and mxapi #####
        try:
            from memryx import mxa  # type:ignore
        except AttributeError:
            raise MxaError(
                message="driver package not installed! Please install the .deb/.rpm/.tgz and try again"
            )

        try:
            from memryx import mxapi # type:ignore
        except AttributeError:
            raise MxaError(
                message="runtime package not installed! Please install the .deb/.rpm/.tgz using `sudo apt install memx-accl` and try again"
            )

        ##### WARNING #####    
        if isinstance(device_ids, int):
            device_ids = [device_ids]
            logger.warning(f"Implicit convert device_id from int to list => {device_ids[0]} to {device_ids}")
        
        if local_mode:
            logger.warning(
                "Running in local mode will lock the device for this process only. "
                "Please switch to shared mode if you get multiple processes to run."
            )
    
        ##### properties #####
        self._device_ids = list(set(device_ids)) # make sure device id are unique
        self._print_lock = threading.Lock()
        self._dfp = self._parse_dfp(dfp)
        self._local_mode = local_mode
        self._chip_gen = 3.1
        self._manager_addr = manager_addr
        self._manager_port = manager_port
        self._ignore_manager = ignore_manager
        
        # specify use_model_shape
        self._use_model_shape = use_model_shape
        if self.sub_version == 1: # sub_version got from _parse_dfp()
            pass
        elif self.sub_version == 0:
            self._use_model_shape = (False,False)

        ##### sanity check #####
        
        for device_id in device_ids:
            if not isinstance(device_id, int) or device_id < 0:
                raise TypeError("All device IDs must be a non-negative integer")

        if local_mode and len(self._device_ids) > 1:
            raise RuntimeError(
                "Python runtime only supports running one device in local mode now."
                "Please switch to shared mode to run multiple devices."
            )
        

        ##### create the dfp runner #####
        self._dfp_runner = DFPRunner(
            dfp_obj=self._dfp,
            manager_addr=self._manager_addr,
            base_port=self._manager_port,
            local_mode=self._local_mode,
            device_ids_to_use=device_ids,
            ignore_manager=self._ignore_manager
        )

        ##### starts dfp runner #####
        if self._local_mode:
            success = self._dfp_runner.init_local()
        else:
            success = self._dfp_runner.init_shared(astuple(scheduler_options), astuple(client_options))

        if not success:
            raise RuntimeError("Init DFP Runner failed!")

        # called at program exit to free resources
        self._finalizer = weakref.finalize(
            self, self._cleanup
        )

        self._configure()

    def get_pressure(self, device_id: int) -> str:
        """
        Get the current pressure status of the specified device.

        Parameters
        ----------
        device_id : int
            The ID of the device for which to retrieve the pressure status. Ignored for local_mode=True, which is one device only.

        Returns
        -------
        str
            The current pressure status of the device. Possible values are "low", "medium", "high" and "full".
        """
        return self._dfp_runner.get_pressure(device_id)
    
    def get_temperature(self, device_id: int) -> float:
        """
        Get the current max temperature of the specified device.

        Parameters
        ----------
        device_id : int
            The ID of the device for which to retrieve the temp. Ignored for local_mode=True, which is one device only.

        Returns
        -------
        float
            The temperature of the device in Celsius.
        """
        return self._dfp_runner.get_temperature(device_id)
    
    def get_avg_power(self, device_id: int) -> float:
        """
        Get the average power of the specified device (if supported).
        NOTE: this is only supported for Shared Mode (local_mode=False) in this SDK release!

        Parameters
        ----------
        device_id : int
            The ID of the device for which to retrieve the power.

        Returns
        -------
        float
            The average power consumption of the device in milliwatts (mW).
        """
        return self._dfp_runner.get_avg_power(device_id)

    def _configure(self, inport_mapping={}, outport_mapping={}):
        for idx_pair, mapping in inport_mapping.items():
            _, port = idx_pair
            for k in ["model_index", "layer_name"]:
                self._dfp.input_ports[port][k] = mapping[k]
        for idx_pair, mapping in outport_mapping.items():
            _, port = idx_pair
            for k in ["model_index", "layer_name"]:
                self._dfp.output_ports[port][k] = mapping[k]

        self._model_idx_to_out_info = defaultdict(lambda: defaultdict(list))
        self._outlayer_to_port_idx = {}
        for i, info in self._dfp.output_ports.items():
            row, col, z, ch = info["shape"]
            if "model_index" in info:
                idx = info["model_index"]
                self._model_idx_to_out_info[idx]["dtypes"].append(np.float32)
                if self._chip_gen > 3:
                    self._model_idx_to_out_info[idx]["shapes"].append([row, col, z, ch])
                else:
                    self._model_idx_to_out_info[idx]["shapes"].append([row, col, ch])
                self._model_idx_to_out_info[idx]["ports"].append(i)
                if "layer_name" in info:
                    layer_name = info["layer_name"]
                    self._model_idx_to_out_info[idx]["layers"].append(layer_name)
                    self._outlayer_to_port_idx[(idx, layer_name)] = i

        self._model_idx_to_in_info = defaultdict(lambda: defaultdict(list))
        self._inlayer_to_port_idx = {}
        for i, info in self._dfp.input_ports.items():
            row, col, z, ch = info["shape"]
            if info["data_type"] in ["float"] or info["data_range_enabled"]:
                dtype = np.float32
            else:
                dtype = np.uint8
            info["dtype"] = dtype
            if "model_index" in info:
                idx = info["model_index"]
                self._model_idx_to_in_info[idx]["dtypes"].append(dtype)
                if self._chip_gen > 3:
                    self._model_idx_to_in_info[idx]["shapes"].append([row, col, z, ch])
                else:
                    self._model_idx_to_in_info[idx]["shapes"].append([row, col, ch])
                self._model_idx_to_in_info[idx]["ports"].append(i)
                if "layer_name" in info:
                    layer_name = info["layer_name"]
                    self._model_idx_to_in_info[idx]["layers"].append(layer_name)
                    self._inlayer_to_port_idx[(idx, layer_name)] = i

        self._create_models()

    def _create_models(self):
        """ In normal cases, this function should be overridden by the subclass """
        # TODO: Replace this function as raise NotImplementedError?
        self._models = []
        for m in self._model_idx_to_in_info:
            self._models.append(
                AcclModel(
                    self._model_idx_to_in_info[m],
                    self._model_idx_to_out_info[m],
                    self,
                    self.input_shape_info,
                    self.output_shape_info,
                )
            )

    def _printl(self, *objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        with self._print_lock:
            print(*objects, sep=sep, end=end, file=file, flush=flush)

    def _parse_dfp(self, dfp) -> Dfp:
        if not isinstance(dfp, Dfp):
            dfp = Dfp(dfp)

        ver = dfp.version
        if ver == "legacy" or int(ver) < 6:
            raise RuntimeError(
                f"Unsupported DFP version: {ver}, please recompile to the latest version"
            )

        self.sub_version = dfp.sub_version

        self.input_shape_info, self.output_shape_info = get_port_info(dfp)

        return dfp

    @property
    def models(self):
        """
        Returns a list of model objects that provide input/output API
        """
        return self._models

    @property
    def input_port_ids(self):
        """
        Returns a list of ids of all active input ports in the accelerator
        """
        all_ports = []
        for model in self._models:
            all_ports.extend(model.input.port_ids)
        return all_ports

    @property
    def output_port_ids(self):
        """
        Returns a list of ids of all active output ports in the accelerator
        """
        all_ports = []
        for model in self._models:
            all_ports.extend(model.output.port_ids)
        return all_ports

    @property
    def mpu_count(self):
        """
        Returns the number of MPUs in the accelerator.
        """
        return self._dfp.num_chips

    @property
    def device_ids(self):
        """
        Returns the device ids defined in the driver
        """
        return self._device_ids

    @property
    def chip_gen(self):
        """
        Returns the architecture generation of the accelerator
        """
        return self._chip_gen

    def outport_assignment(self, model_idx=0):
        """
        Returns a dictionary which maps output port ids to model output layer names for
        the model specified by `model_idx`

        Parameters
        ----------
        model_idx: int
            Index of the model whose output port assignment is returned
        """
        self._ensure_valid_model_idx(model_idx)
        return self._models[model_idx].output.port_assignment

    def inport_assignment(self, model_idx=0):
        """
        Returns a dictionary which maps input port ids to model input layer names for
        the model specified by `model_idx`

        Parameters
        ----------
        model_idx: int
            Index of the model whose input port assignment is returned
        """
        self._ensure_valid_model_idx(model_idx)
        return self._models[model_idx].input.port_assignment

    def set_preprocessing_model(self, model_or_path, model_idx=0):
        """
        Supply the path to a model/file that should be run to pre-process the input feature map.
        This is an optional feature that can be used to automatically run the pre-processing model
        output by the NeuralCompiler

        .. note::

            This function currently does not support PyTorch models

        .. warning::

            This function is currently not available on the ARM platform

        Parameters
        ----------
        model_or_path: obj or str
            Can be either an already loaded model such as a tf.keras.Model object for Keras,
            or a str path to a model file.
        model_idx: int
            Index of the model on the accelerator whose input feature map should be
            pre-processed by the supplied model
        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].input.set_preprocessing(model_or_path)

    def set_postprocessing_model(self, model_or_path, model_idx=0):
        """
        Supply the path to a model/file that should be run to post-process the output feature maps
        This is an optional feature that can be used to automatically run the post-processing model
        output by the NeuralCompiler

        .. note::

            This function currently does not support PyTorch models

        .. warning::

            This function is currently not available on the ARM platform

        Parameters
        ----------
        model_or_path: obj or str
            Can be either an already loaded model such as a tf.keras.Model object for Keras,
            or a string path to a model file.
        model_idx: int
            Index of the model on the accelerator whose output should be
            post-processed by the supplied model

        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].output.set_postprocessing(model_or_path)

    def shutdown(self):
        """
        Shutdown the accelerator to make it available for other processes to use.
        If this function is not called while the current program is running,
        other processes will not be able to access the same accelerator until
        the current program terminates.
        """
        self._dfp_runner.shutdown()

        del self._dfp
        self._dfp = None

        time.sleep(1) # wait for mx server to unlock the device

    def _ensure_valid_model_idx(self, model_idx):
        if model_idx not in range(len(self._models)):
            raise IndexError(
                f"Valid model indices are in the range [0, {len(self._models) - 1}], "
                f"but got: {model_idx}"
            )

    def _ensure_nd_array(self, data, model_idx):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Model {model_idx} expected np.ndarray, got: {type(data)}")

    def _ensure_seq_nd_array(self, data, model_idx):
        for item in data:
            if not isinstance(item, np.ndarray):
                raise TypeError(
                    f"Model {model_idx} expected np.ndarray, got: {type(data)}"
                )

    def _ensure_input_len(self, data, target_len, model_idx):
        if len(data) != target_len:
            raise RuntimeError(
                f"Model {model_idx} has {target_len} inputs, "
                f"but data is a sequence of length: {len(data)}"
            )

    def _cleanup(self):
        self.shutdown()


class AcclModel(ABC):
    def __init__(
        self,
        input_info,
        output_info,
        accl,
        input_shape_info,
        output_shape_info,
        use_model_shape=(True,True),
    ):
        self._accl = accl
        
        
        # cond var used in in/out workers for syncrhonization
        self.cond_flush = threading.Condition()
        self.cond_flush_flag = False

        # cond var used in in/out workers for syncrhonization
        self.cond_output = threading.Condition()
        self.cond_output_flag = False
        
        self.input = AcclModelInput(input_info, accl, self, input_shape_info, use_model_shape[0])
        self.output = AcclModelOutput(
            output_info, accl, self, output_shape_info, use_model_shape[1]
        )


class AcclModelInput(ABC):
    def __init__(
        self, info, accl, model, input_shape_info, use_model_shape_in
    ):
        self._info = info
        self._accl = accl
        self._model = model
        self._pre_model = _PassThroughModel()
        self._input_shape_info = input_shape_info
        self._use_model_shape_in = use_model_shape_in
        
        self.send_cnt = 0 # number of input frames sent to the accelerator, for debug

    @property
    def port_assignment(self):
        """
        Returns a dictionary which maps port ids to output layer names
        """
        return {
            self._info["ports"][i]: self._info["layers"][i]
            for i in range(len(self._info["ports"]))
        }

    @property
    def port_count(self):
        """
        Returns the number of input ports used by the model
        """
        return len(self._info["ports"])

    @property
    def port_ids(self):
        """
        Returns a list of input port ids used by the model
        """
        return self._info["ports"]

    @property
    def port_spec(self):
        """
        Returns a list of (shape, dtype) tuples for each input port used by the model
        """
        return [
            (shape, dtype)
            for shape, dtype in zip(self._info["shapes"], self._info["dtypes"])
        ]

    def set_preprocessing(self, model):
        if model is None:
            raise RuntimeError("pre-processing model is None")

        if not self._use_model_shape_in and self._accl.sub_version == 1:
            raise RuntimeError(
                "The --use_model_shape_in flag must be set to True when set_preprocessing is called."
            )

        if isinstance(model, (str, Path)):
            self._pre_model = ModelLoader().load(str(model))
        else:
            self._pre_model = ModelLoader().from_memory(model)

        input_batch_sizes = [shape[0] for shape in self._pre_model.input_shapes()]
        if not all(B == 1 or not isinstance(B, int) for B in input_batch_sizes):
            raise RuntimeError(
                (
                    "Pre-processing model requires singleton or variable batch dimension "
                    f"but has input batch sizes of {input_batch_sizes}. "
                    "You may handle pre-processing manually or redefine the model."
                )
            )

    def _stream_ifmaps(self, ifmaps):
        if not isinstance(ifmaps, (list, tuple)):
            ifmaps = [ifmaps]
        model_idx = self._accl.models.index(self._model)
        if len(ifmaps) != len(self._info["ports"]):
            raise RuntimeError(
                f"Model {model_idx} expects {len(self._info['ports'])} inputs, but input function returned {len(ifmaps)} frames"
            )
        
        # Start streaming
        for fmap, port, shape, dtype in zip(
            ifmaps,
            self._info["ports"],
            self._info["shapes"],
            self._info["dtypes"],
        ):
            if len(shape) == 4 and shape[-2] == 1 and len(list(fmap.shape)) == 3:
                shape = shape[:-2] + [shape[-1]]
            if self._use_model_shape_in is True:
                fmap = convert_to_mxainput(self._input_shape_info[port], fmap)
            compressed_shape = [i for i in shape if i != 1]
            if list(np.squeeze(fmap).shape) not in [compressed_shape, list(compressed_shape)]:
                raise RuntimeError(
                    f"Model {model_idx} input port {port} expects data of shape {shape}, but got {fmap.shape}"
                )
            if list(fmap.shape) == [1] + list(shape):
                fmap = np.squeeze(fmap, 0)
            if fmap.dtype != dtype:
                raise RuntimeError(
                    f"Model {model_idx} input port {port} expects data of type {dtype}, but got {fmap.dtype}"
                )

            logger.debug(f"Model {model_idx} send ifmap to port {port} with shape {fmap.shape}")
            
            # NOTE: explicitly copy the fmap because
            # we cannot trust the user will leave the reference to 'f'
            # alone while the accelerator works on it. Perhaps we can add
            # an 'immutable' flag in the future to avoid this copy.
            success = self._accl._dfp_runner.stream_ifmap(model_idx, port, fmap.copy())
            if not success:
                raise RuntimeError(f"Driver stream input function error")

        # increment counter for debug
        self.send_cnt += 1
        logger.debug(f"Model {model_idx} successfully send ifmap. Total send cnt: {self.send_cnt}")


class AcclModelOutput(ABC):
    def __init__(
        self, info, accl, model, output_shape_info, use_model_shape_out
    ):
        self._info = info
        self._accl = accl
        self._model = model
        self._post_model = _PassThroughModel()
        self._output_shape_info = output_shape_info
        self._use_model_shape_out = use_model_shape_out
        self.recv_cnt = 0  # number of output frames received from the accelerator, for debug
        
    @property
    def port_assignment(self):
        """
        Returns a dictionary which maps port ids to model output layer names
        """
        return {
            self._info["ports"][i]: self._info["layers"][i]
            for i in range(len(self._info["ports"]))
        }

    @property
    def port_count(self):
        """
        Returns the number of output ports used by the model
        """
        return len(self._info["ports"])

    @property
    def port_ids(self):
        """
        Returns a list of output port ids used by the model
        """
        return self._info["ports"]

    @property
    def port_spec(self):
        """
        Returns a list of (shape, dtype) tuples for each output port used by the model
        """
        shapes = []
        for shape in self._info["shapes"]:
            shapes.append(shape)
        return [(shape, dtype) for shape, dtype in zip(shapes, self._info["dtypes"])]

    def set_postprocessing(self, model):
        if model is None:
            raise RuntimeError("post-processing model is None")

        if not self._use_model_shape_out and self._accl.sub_version == 1:
            raise RuntimeError(
                "The --use_model_shape_out flag must be set to True when set_postprocessing is called."
            )

        if isinstance(model, (str, Path)):
            self._post_model = ModelLoader().load(str(model))
        else:
            self._post_model = ModelLoader().from_memory(model)

        input_batch_sizes = [shape[0] for shape in self._post_model.input_shapes()]
        if not all(B == 1 or not isinstance(B, int) for B in input_batch_sizes):
            raise RuntimeError(
                (
                    "Post-processing model requires singleton or variable batch dimension "
                    f"but has input batch sizes of {input_batch_sizes}. "
                    "You may handle post-processing manually or redefine the model."
                )
            )

    def _stream_ofmaps(self, model_idx, outputs, outputs_by_name, timeout=500):

        outputs.clear()
        outputs_by_name.clear()

        for idx, port in enumerate(self._info["ports"]):
            shape = self._info["shapes"][idx]
            dtype = self._info["dtypes"][idx]
            ofmap = np.zeros(shape, dtype=dtype)

            logger.debug(f"Model {model_idx} recv ofmap from port {port} with shape {ofmap.shape}")

            # stream output feature map
            success = self._accl._dfp_runner.stream_ofmap(model_idx, port, ofmap)
            if not success:
                return False

            # make sure ofmap has the correct shape and dtype
            if self._use_model_shape_out is True:
                ofmap = convert_to_modeloutput(self._output_shape_info[port], ofmap)
            elif self._use_model_shape_out is False and self._accl.sub_version == 0:
                if len(ofmap.shape) == 4 and ofmap.shape[-2] == 1:
                    ofmap = np.squeeze(ofmap, -2)

            # stack output feature maps
            outputs.append(ofmap)
            outputs_by_name[self._info["layers"][idx]] = ofmap

        # increment counter for debug
        self.recv_cnt += 1
        logger.debug(f"Model {model_idx} successfully recv ofmap. Total recv cnt: {self.recv_cnt}")

        # successfully streamed all output feature maps
        return True


class SyncAccl(Accl):
    """
    This class provides a synchronous API for the MemryX hardware accelerator, which performs input and output
    sequentially per model. The accelerator is abstracted as a collection of models. You can select
    the desired model specifying its index to the member function.

    Parameters
    ----------
    dfp : bytes or str
        Path to the DFP file generated by the NeuralCompiler, or a byte array
        representing the DFP content.

    device_ids : list of int, optional
        List of MemryX device IDs to be used for executing the DFP.
        Default is [0].

    use_model_shape : tuple of bool, optional
        Tuple in the form (input_shape, output_shape). Specifies whether to enforce
        the original model input/output shapes (True) or use MXA runtime shapes (False).
        Default is (True, True).

    local_mode : bool, optional
        If True, executes the DFP in local mode, which can improve performance for
        single-process use. Incompatible with multi-DFP and multi-process use.
        Default is False.

    scheduler_options : SchedulerOptions, optional
        Scheduler configuration corresponding to the SchedulerOptions struct:
        
        - frame_limit : int  
          Number of frames to process before swapping out. Default 20.
        - time_limit : int  
          Maximum idle time (in milliseconds) before swapping out. Default 250.
        - stop_on_empty : bool  
          Whether to immediately swap if the input queue is empty. Default False.
        - ifmap_queue_size : int  
          Size of the shared input feature map queue. Default 16.
        - ofmap_queue_size : int  
          Size of the per-client output feature map queues. Default 21.

    client_options : ClientOptions, optional
        Client runtime behavior configuration, corresponding to the ClientOptions struct:

        - smoothing : bool  
          Whether to enable FPS smoothing. Default False.
        - fps_target : float  
          Target frames per second. A delay of 1 / fps_target seconds is enforced 
          between frames. Default 0.

    manager_addr : str, optional
        Path to the mxa-manager socket. Needed in Docker or managed environments.
        Default is "/run/mxa_manager/".

    manager_port : int, optional
        Port for mxa-manager connection (used primarily in containerized deployments).
        Default is 10000.

    ignore_manager : bool, optional
        If True, bypasses the manager and forces local mode. May cause crashes if
        multiple containers/processes attempt to use the same device.
        Default is False.

    Warnings
    --------
    Setting `ignore_manager` to True disables coordination with other processes and may result 
    in device conflicts if multiple clients or containers access the same device concurrently. 
    Use with caution.

    Examples
    --------

    .. code-block:: python

        import tensorflow as tf
        import numpy as np
        from memryx import NeuralCompiler, SyncAccl

        # Compile a MobileNet model for testing.
        # Typically, comilation need to be done one time only.
        model = tf.keras.applications.MobileNet()
        nc = NeuralCompiler(models=model)
        dfp = nc.run()

        # Prepare the input data
        img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        data = tf.keras.applications.mobilenet.preprocess_input(img)

        # Accelerate using the MemryX hardware
        accl = SyncAccl(dfp)
        outputs = accl.run(data)  # Run sequential acceleration on the input data.

        # Explicitly free up hardware
        accl.shutdown()

    .. warning::

        MemryX accelerator is a streaming processor that the user can supply with pipelined input data. Using the synchronous API to perform sequential execution of multiple input frames may result in a significant performance penalty. The user is advised to use the send/receive functions on separate threads or to use the asynchronous API interface.
    """

    # TODO: inport_mapping and outport_mapping will get removed once the DFP branch gets merged

    def __init__(
        self,
        dfp: Union[str, Path, bytes, Dfp],
        device_ids: Union[List[int], int]=[0],
        use_model_shape: Tuple[bool, bool]=(True, True),
        local_mode: bool = False,
        scheduler_options: SchedulerOptions =  SchedulerOptions(),
        client_options: ClientOptions = ClientOptions(),
        manager_addr: str = "/run/mxa_manager/",
        manager_port: int = 10000,
        ignore_manager: bool = False,
    ):
        super().__init__(dfp, device_ids, use_model_shape, local_mode, scheduler_options, client_options, manager_addr, manager_port, ignore_manager)

    def _create_models(self):
        self._models: list[SyncAcclModel] = []
        for m in sorted(self._model_idx_to_in_info):
            self._models.append(
                SyncAcclModel(
                    self._model_idx_to_in_info[m],
                    self._model_idx_to_out_info[m],
                    self,
                    self.input_shape_info,
                    self.output_shape_info,
                    use_model_shape=self._use_model_shape,
                )
            )

    def send(self, data, model_idx=0, timeout=None):
        """
        Sends input data to the accelerator for the specified model.

        For the model identified by `model_idx`, this method transfers the input data
        to the accelerator by copying it into the model's input buffer(s). If the
        input buffer(s) are full, the call blocks according to the value of `timeout`.

        Parameters
        ----------
        data : np.ndarray or sequence of np.ndarray
            The input data to be transferred. This is typically the preprocessed input
            array (or list/tuple of arrays) expected by the model. For multi-input models,
            a sequence of arrays must be provided, with each array corresponding to an model input.

        model_idx : int, optional
            Index of the model to which the data should be sent.
            Default is 0.

        timeout : int or None, optional
            The maximum time in milliseconds to block if the input buffer is full.
            If `None` (default), the call blocks indefinitely until space becomes available.
            If a positive integer is provided, the call blocks for up to `timeout` milliseconds,
            after which it raises a `TimeoutError` if space is still unavailable.

        Raises
        ------
        TimeoutError
            Raised if the input buffer does not become available within the specified `timeout`.
     
        """
        m = model_idx
        self._ensure_valid_model_idx(m)

        data = self.__preprocess(data, m)
        if not isinstance(data, (list, tuple)):
            data = [data]
        self._ensure_seq_nd_array(data, m)

        inports = self._models[m].input.port_ids
        self._ensure_input_len(data, len(inports), m)
        for i in range(len(data)):
            self._models[m].input.send(data[i], inports[i], m, timeout=timeout)

    def receive(self, model_idx=0, timeout=None):
        """
        Receives output data from the accelerator for the specified model.

        For the model identified by `model_idx`, this method retrieves data from the
        accelerator's output buffer(s). If no output data is currently available, the
        call will block according to the specified `timeout` policy.

        Parameters
        ----------
        model_idx : int, optional
            Index of the model from which output data should be retrieved.
            Default is 0.

        timeout : int or None, optional
            Maximum number of milliseconds to wait if no output is immediately available.
            If `None` (default), blocks indefinitely until output data becomes available.
            If a positive integer is provided, the call will block for up to `timeout`
            milliseconds before raising a `TimeoutError`.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            The output data retrieved from the model. For single-output models, a single
            `np.ndarray` is returned. For multi-output models, a tuple of arrays is returned,
            with one entry per output of the specified model.

        Raises
        ------
        TimeoutError
            Raised if no output data becomes available within the specified `timeout`.
       
        Example
        -------

        .. code-block:: python

            from tensorflow import keras
            import numpy as np
            from memryx import NeuralCompiler, SyncAccl

            def generate_frame():
                # Prepare the input data
                img = np.random.rand(1, 224, 224, 3).astype(np.float32)
                return keras.applications.mobilenet.preprocess_input(img)

            # Compile a MobileNet model for testing.
            # Typically, comilation need to be done one time only.
            model = keras.applications.MobileNet()
            nc = NeuralCompiler(models=model)
            dfp = nc.run()

            # Accelerate using the MemryX hardware
            accl = SyncAccl(dfp)

            frame_count = 10
            send_count, recv_count = 0, 0
            outputs = []
            while recv_count < frame_count:
                if send_count < frame_count:
                    accl.send(generate_frame())
                    send_count += 1
                try:
                    output = accl.receive(timeout=1)
                    outputs.append(output)
                except TimeoutError:
                    continue  # try sending the next frame if output is not ready yet
                recv_count += 1

            # Explicitly free up hardware
            accl.shutdown()

        """
        m = model_idx
        self._ensure_valid_model_idx(m)
        outputs = []
        outputs_by_name = {}

        model_output = self._models[m].output
        outports = model_output.port_ids
        output_count = len(outports)

        for i in range(output_count):
            ofmap = model_output.receive(outports[i], m, timeout=timeout)
            outputs.append(ofmap)
            outputs_by_name[model_output._info["layers"][i]] = ofmap

        outputs = self.__postprocess(outputs_by_name, m)

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def run(self, inputs, model_idx=0):
        """
        Sends input data to the specified model and retrieves the corresponding outputs.

        This method combines `send()` and `receive()` into a single sequential operation.
        It transfers the provided input(s) to the accelerator and blocks until the
        corresponding outputs are ready.

        Parameters
        ----------
        inputs : np.ndarray or list of np.ndarray or list of list of np.ndarray
            The preprocessed input data for the model.
            - A single `np.ndarray` is used for single-input models.
            - A list of `np.ndarray` is used for multi-input models.
            - A nested list of inputs (e.g., list of list of arrays) can be used to batch multiple input sets for better throughput.
            Each individual `np.ndarray` must match the input shape and data type
            expected by the model.

            Note: Stacking multiple input sets into a single `np.ndarray` is **not supported**.

        model_idx : int, optional
            Index of the model to which the inputs should be sent.
            Default is 0.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            The model's output data. A single `np.ndarray` is returned for models
            with one output, or a tuple of arrays for multi-output models.
       """
        self._ensure_valid_model_idx(model_idx)

        data = inputs
        stacked_data = []
        if not isinstance(data, (list, tuple)):
            if not isinstance(data, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got: {type(data)}")
            stacked_data = [[data]]  # 1 set, 1 input
        else:
            for item in data:
                if isinstance(item, np.ndarray):
                    stacked_data.append([item])
                elif not isinstance(item, (list, tuple)):
                    raise TypeError(
                        f"Expected sequence of np.ndarray, got: {type(item)}"
                    )
                else:
                    for inner_item in item:
                        if isinstance(inner_item, np.ndarray):
                            continue
                        raise TypeError(f"Expected np.ndarray, got: {type(item)}")
                    stacked_data.append(item)

        all_outputs = self.__threaded_run(stacked_data, model_idx)

        unstacked_outputs = []
        for item in all_outputs:
            if len(item) == 1:
                unstacked_outputs.append(item[0])
            else:
                unstacked_outputs.append(item)

        if len(unstacked_outputs) == 1:
            return unstacked_outputs[0]

        return unstacked_outputs

    def shutdown(self):
        """
        Shutdown the accelerator to make it available for other processes to use.
        If this function is not called while the current program is running,
        other processes will not be able to access the same accelerator until
        the current program terminates.
        Note that after calling shutdown, a new instance of SyncAccl must be created
        to re-initialize the accelerator, before it can be run again from the current
        program.
        """
        super().shutdown()

    def __threaded_run(self, stacked_data, model_idx):
        self.cond_recv = threading.Condition()
        self.send_counter = 0
        self.recv_counter = 0

        exc = []
        sender = threading.Thread(
            target=self.__send, args=(stacked_data, model_idx, exc), daemon=True
        )
        all_outputs = []
        receiver = threading.Thread(
            target=self.__receive,
            args=(all_outputs, len(stacked_data), model_idx, exc),
            daemon=True,
        )
        self._running = True
        receiver.start()
        sender.start()
        time.sleep(0.01)
        try:
            receiver.join()
            sender.join()
        except KeyboardInterrupt:
            logger.critical("Terminating run due to KeyboardInterrupt")
            self._running = False
            receiver.join()
            sender.join()
        if exc:
            raise exc[0] from None
        return all_outputs

    def __send(self, stacked_data, model_idx, exc):
        i = 0
        while self._running and i < len(stacked_data):
            try:
                data = stacked_data[i]
                self.send(data, model_idx)

                # notify output thread here comes a new ifmap
                with self.cond_recv:
                    self.send_counter += 1
                    self.cond_recv.notify()

            except Exception as e:
                self._running = False
                logger.critical(f"Terminating run due to exception in send: {str(e)}")
                exc.append(e)
                return
            i += 1

    def __receive(self, outputs, out_count, model_idx, exc):

        i = 0
        while self._running and i < out_count:

            with self.cond_recv:
                if self.send_counter <= self.recv_counter:
                    # if we got here, it means SyncAccl is still running, but haven't sent any new ifmap
                    # So wait for it patiently
                    self.cond_recv.wait_for(lambda: self.send_counter > self.recv_counter, timeout=0.5)
                    continue

            try:
                out = self.receive(model_idx)

                with self.cond_recv:
                    self.recv_counter += 1

            except TimeoutError:
                continue
            except Exception as e:
                exc.append(e)
                return
            if not isinstance(out, list):
                out = [out]
            outputs.append(out)
            i += 1

    def __preprocess(self, data, model_idx):
        try:
            model_input = self._models[model_idx].input
            data = model_input._pre_model.predict_for_mxa(
                data,
                model_input._use_model_shape_in,
                pre_model=True,
                output_order=model_input._info["layers"],
            )
        except Exception:
            raise RuntimeError(
                "Failed to run inference on the pre-processing model\n"
                "Ensure that the pre-processing model passed to `set_preprocessing_model()` "
                "and the model in the DFP come from the same model passed to the NeuralCompiler"
            ) from None
        return data

    def __postprocess(self, outputs_by_name, model_idx):
        try:
            model_output = self._models[model_idx].output
            fmaps = model_output._post_model.predict_for_mxa(
                outputs_by_name, model_output._use_model_shape_out
            )
        except Exception:
            raise RuntimeError(
                "Failed to run inference on the post-processing model\n"
                "Ensure that the post-processing model passed to `set_postprocessing_model()` "
                "and the model in the DFP come from the same model passed to the NeuralCompiler"
            ) from None
        return fmaps


def cleanup(func):
    def cleanup_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self = args[0]
            self._accl._cleanup()
            raise e from None

    return cleanup_wrapper


class SyncAcclModel(AcclModel):
    def __init__(
        self,
        input_info,
        output_info,
        accl,
        input_shape_info,
        output_shape_info,
        use_model_shape=(True,True),
    ):
        super().__init__(
            input_info, output_info, accl, input_shape_info, output_shape_info
        )
        self.input = SyncAcclModelInput(
            input_info, accl, self, input_shape_info, use_model_shape[0]
        )
        self.output = SyncAcclModelOutput(
            output_info, accl, self, output_shape_info, use_model_shape[1]
        )


class SyncAcclModelInput(AcclModelInput):
    def __init__(
        self, info, accl, model, input_shape_info, use_model_shape_in=True
    ):
        super().__init__(
            info,
            accl,
            model,
            input_shape_info,
            use_model_shape_in=use_model_shape_in,
        )
        self._input_shape_info = input_shape_info

    def send(self, data, port_idx, model_idx, timeout=None):
        if not isinstance(port_idx, int):
            raise TypeError(f"Expected port_id to be an int but got: {type(port_idx)}")

        if port_idx not in self._info["ports"]:
            raise ValueError(f"Input port with idx: {port_idx} is inactive")

        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected data to be a `np.ndarray` but got: {type(data)}")

        if timeout is not None and not isinstance(timeout, numbers.Real):
            raise TypeError(
                f"Expected timeout to be a real number but got: {type(timeout)}"
            )

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")

        if timeout is None:
            timeout = 0

        if self._use_model_shape_in is True:
            data = convert_to_mxainput(self._input_shape_info[port_idx], data)

        relative_port_idx = port_idx - min(self._info["ports"])

        shape = list(data.shape)
        reqd_shape = list(self._info["shapes"][relative_port_idx])
        if len(reqd_shape) == 4 and reqd_shape[-2] == 1 and len(shape) == 3:
            reqd_shape = reqd_shape[:-2] + [reqd_shape[-1]]
        if shape not in [reqd_shape, [1] + reqd_shape]:
            raise RuntimeError(
                f"Port at index: {port_idx} expects data of shape: {reqd_shape}, "
                f"but got: {shape}"
            )
        reqd_dtype = self._info["dtypes"][relative_port_idx]
        if reqd_dtype != data.dtype:
            raise TypeError(
                f"Port at index: {port_idx} expects data of type: {reqd_dtype}, "
                f"but got: {data.dtype}"
            )

        # NOTE: explicitly copy the 'data' because we cannot
        # trust the user will leave the reference to 'data' alone while the
        # accelerator works on it. Perhaps we can add an 'immutable' flag in
        # the future to avoid this copy.
        success = self._accl._dfp_runner.stream_ifmap(model_idx, port_idx, data.copy())
        if not success:
            raise RuntimeError(f"Driver stream input function error for port idx: {port_idx}")

        # increment counter for debug
        self.send_cnt += 1
        logger.debug(f"Model {model_idx} successfully send ifmap. Total send cnt: {self.send_cnt}")


class SyncAcclModelOutput(AcclModelOutput):
    def __init__(
        self, info, accl, model, output_shape_info, use_model_shape_out=True
    ):
        super().__init__(
            info,
            accl,
            model,
            output_shape_info,
            use_model_shape_out=use_model_shape_out,
        )
        self._output_shape_info = output_shape_info

    def receive(self, port_idx, model_idx, timeout=None):
        if timeout is not None and not isinstance(timeout, numbers.Real):
            raise TypeError(
                f"Expected timeout to be a real number but got: {type(timeout)}"
            )

        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")

        if not isinstance(port_idx, int):
            raise TypeError(f"Expected port_idx to be an int but got: {type(port_idx)}")

        if port_idx not in self._info["ports"]:
            raise ValueError(f"Output port with idx: {port_idx} is inactive")

        if timeout is None:
            timeout = 0

        relative_port_idx = port_idx - min(self._info["ports"])

        shape = self._info["shapes"][relative_port_idx]
        ofmap = np.zeros(shape, dtype=np.float32)
        
        # stream output feature map
        success = self._accl._dfp_runner.stream_ofmap(model_idx, port_idx, ofmap)
        if not success:
            raise TimeoutError(
                f"Driver stream output function timed out for port at idx: {port_idx}"
            )

        # make sure ofmap has the correct shape and dtype
        if self._use_model_shape_out is True:
            ofmap = convert_to_modeloutput(self._output_shape_info[port_idx], ofmap)
        elif self._use_model_shape_out is False and self._accl.sub_version == 0:
            if len(ofmap.shape) == 4 and ofmap.shape[-2] == 1:
                ofmap = np.squeeze(ofmap, -2)
        
        # increment counter for debug
        self.recv_cnt += 1
        logger.debug(f"Model {model_idx} successfully recv ofmap. Total recv cnt: {self.recv_cnt}")
        
        return ofmap

    
class AsyncAccl(Accl):
    """
    This class provides an asynchronous API to run models on the MemryX hardware accelerator.
    The user provides callback functions to feed data and receive outputs from the accelerator,
    which are then called whenever a model is ready to accept/output data.
    This pipelines execution of the models and allows the accelerator to run at full speed.

    Parameters
    ----------
    dfp : bytes or str
        Path to the DFP file generated by the NeuralCompiler, or a byte array
        representing the DFP content.

    device_ids : list of int, optional
        List of MemryX device IDs to be used for executing the DFP.
        Default is [0].

    use_model_shape : tuple of bool, optional
        Tuple in the form (input_shape, output_shape). Specifies whether to enforce
        the original model input/output shapes (True) or use MXA runtime shapes (False).
        Default is (True, True).

    local_mode : bool, optional
        If True, executes the DFP in local mode, which can improve performance for
        single-process use. Incompatible with multi-DFP and multi-process use.
        Default is False.

    scheduler_options : SchedulerOptions, optional
        Scheduler configuration corresponding to the SchedulerOptions struct:
        
        - frame_limit : int  
          Number of frames to process before swapping out. Default 20.
        - time_limit : int  
          Maximum idle time (in milliseconds) before swapping out. Default 250.
        - stop_on_empty : bool  
          Whether to immediately swap if the input queue is empty. Default False.
        - ifmap_queue_size : int  
          Size of the shared input feature map queue. Default 16.
        - ofmap_queue_size : int  
          Size of the per-client output feature map queues. Default 21.

    client_options : ClientOptions, optional
        Client runtime behavior configuration, corresponding to the ClientOptions struct:

        - smoothing : bool  
          Whether to enable FPS smoothing. Default False.
        - fps_target : float  
          Target frames per second. A delay of 1 / fps_target seconds is enforced 
          between frames. Default 0.

    manager_addr : str, optional
        Path to the mxa-manager socket. Needed in Docker or managed environments.
        Default is "/run/mxa_manager/".

    manager_port : int, optional
        Port for mxa-manager connection (used primarily in containerized deployments).
        Default is 10000.

    ignore_manager : bool, optional
        If True, bypasses the manager and forces local mode. May cause crashes if
        multiple containers/processes attempt to use the same device.
        Default is False.
    
    Warnings
    --------
    Setting `ignore_manager` to True disables coordination with other processes and may result 
    in device conflicts if multiple clients or containers access the same device concurrently. 
    Use with caution.


    Examples
    --------

    .. code-block:: python

        from tensorflow import keras
        import numpy as np
        from memryx import NeuralCompiler, AsyncAccl

        # Define the callback that will return model input data
        def data_source():
            for i in range(10):
                img = np.random.rand(1, 224, 224, 3).astype(np.float32)
                data = keras.applications.mobilenet.preprocess_input(img)
                yield data

        # Define the callback that will process the outputs of the model
        outputs = []

        def output_processor(*logits):
            logits = logits[0]
            preds = keras.applications.mobilenet.decode_predictions(logits)
            outputs.append(preds)

        # Compile a MobileNet model for testing.
        # Typically, comilation need to be done one time only.
        model = keras.applications.MobileNet()
        nc = NeuralCompiler(models=model)
        dfp = nc.run()

        # Accelerate using the MemryX hardware
        accl = AsyncAccl(dfp)

        # Starts asynchronous execution of input generating callback
        accl.connect_input(data_source)

        # Starts asynchronous execution of output processing callback
        accl.connect_output(output_processor)

        # Wait for the accelerator to finish execution
        accl.wait()

        # Explicitly free up hardware
        accl.shutdown()

    """

    def __init__(
        self,
        dfp: Union[str, Path, bytes, Dfp],
        device_ids: Union[List[int], int]=[0],
        use_model_shape: Tuple[bool, bool]=(True, True),
        local_mode: bool = False,
        scheduler_options: SchedulerOptions =  SchedulerOptions(),
        client_options: ClientOptions = ClientOptions(),
        manager_addr: str = "/run/mxa_manager/",
        manager_port: int = 10000,
        ignore_manager: bool = False,
    ):
        self.use_model_shape = use_model_shape
        super().__init__(dfp, device_ids, use_model_shape, local_mode, scheduler_options, client_options, manager_addr, manager_port, ignore_manager)

    def _create_models(self):
        self._models = []
        for m in sorted(self._model_idx_to_in_info):
            self._models.append(
                AsyncAcclModel(
                    self._model_idx_to_in_info[m],
                    self._model_idx_to_out_info[m],
                    self,
                    self.input_shape_info,
                    self.output_shape_info,
                    use_model_shape=self._use_model_shape,
                )
            )

    def connect_input(self, callback, model_idx=0):
        """
        Sets a callback function to execute when the accelerator is ready to begin
        processing an input frame for the specified model.

        Parameters
        ----------
        callback : callable
            A function or bound method that is invoked asynchronously when the accelerator
            signals readiness to process a new input frame for the model identified by `model_idx`.

            - The `callback` must take no arguments.
            - It must return either a single `np.ndarray` (for single-input models) or a
              sequence of `np.ndarray` objects (for multi-input models).
            - The returned arrays must match the data types and shapes expected by the model.

            Returning `None` or raising an exception from the callback signals the end
            of the input stream for the model.

            If a preprocessing model was configured via `set_preprocessing`, the output
            of `callback` will first be passed through that model before being sent to
            the accelerator.

        model_idx : int, optional
            Index of the target model to which the input callback should be bound.
            Default is 0.

        Raises
        ------
        RuntimeError: If the signature of `callback` contains any paramters
        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].input.connect(callback)

    def connect_output(self, callback, model_idx=0):
        """
        Sets a callback function to execute when the outputs of the specified model are ready.

        Parameters
        ----------
        callback : callable
            A function or bound method that is invoked asynchronously when the accelerator
            finishes processing an input frame for the model identified by `model_idx`.

            - The arguments to `callback` must exactly match the number and order of output
              feature maps as defined by the model's port configuration (retrievable via
              `outport_assignment`).
            - No additional parameters beyond the model outputs are allowed in the function signature.

            If a post-processing model was previously set using `set_postprocessing`,
            the model's raw outputs will first be passed through that post-processing model,
            and the resulting outputs will be passed to `callback` instead.

        model_idx : int, optional
            Index of the model whose outputs should be routed to `callback`.
            Default is 0.

        """
        self._ensure_valid_model_idx(model_idx)
        self._models[model_idx].output.connect(callback)

    def stop(self):
        """
        Send a signal to stop each of the models running on the accelerator.
        This call blocks until each of the models stops and cleans up its
        resources.
        """
        for model in self._models:
            model.stop()
        time.sleep(0.01)

    def wait(self):
        """
        Make the main thread wait for the accelerator to finish executing all models.

        Raises
        ------
        RuntimeError: If the any of the model's inputs/outputs are left unconnected
        """
        for i, model in enumerate(self._models):
            if not model.input.connected():
                raise RuntimeError(
                    f"Model {i}'s input is not connected, "
                    f"please call `connect_input(f, {i})` "
                    "where f is the callback function that feeds data to Model {i}"
                )
            if not model.output.connected():
                raise RuntimeError(
                    f"Model {i}'s output is not connected, "
                    f"please call `connect_output(f, {i})` "
                    "where f is the callback function that "
                    "consumes data output by the Model {i}"
                )

        try:
            for model in self._models:
                model.wait()
        except KeyboardInterrupt:
            for model in self._models:
                model.stop()

    def shutdown(self):
        """
        Stop all currently running models and shutdown the accelerator
        to make it available for other processes to use. Calling stop() only
        stops the running models, but doesn't allow other processes to use
        the same accelerator while the current program is running.
        Note that after calling shutdown, a new instance of AsyncAccl must be
        created to re-initialize the accelerator, before it can be run again
        from the same program.
        """
        self.stop()
        super().shutdown()

    def _cleanup(self):
        self.stop()
        super()._cleanup()


class AsyncAcclModel(AcclModel):
    def __init__(
        self,
        input_info,
        output_info,
        accl,
        input_shape_info,
        output_shape_info,
        use_model_shape=(True,True),
    ):
        super().__init__(
            input_info, output_info, accl, input_shape_info, output_shape_info
        )

        self.input = AsyncAcclModelInput(
            input_info, accl, self, input_shape_info, use_model_shape[0]
        )
        self.output = AsyncAcclModelOutput(
            output_info, accl, self, output_shape_info, use_model_shape[1]
        )

    def wait(self):
        self.input.wait()
        self.output.wait()

    def stop(self):
        self.input.stop()
        self.output.stop()


def graceful_shutdown(func):
    def exc_handling_wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            buf = io.StringIO()
            traceback.print_exc(file=buf)
            logger.critical(
                buf.getvalue()
            )  # log exception now as other model.stop() may block forever
            buf.close()

            self = args[0]
            self._stop_event.set()

            # notify output thread to flush out ofmaps left in the driver output queue
            logger.debug(f"Notify output thread to flush out ofmaps")
            if hasattr(self, '_model') and hasattr(self._model, 'cond_flush'):
                with self._model.cond_flush:
                    self._model.cond_flush_flag = True
                    self._model.cond_flush.notify()

            raise e from None

    return exc_handling_wrapper


class AsyncAcclModelInput(AcclModelInput):
    def __init__(
        self, info, accl, model, input_shape_info, use_model_shape_in=True
    ):
        super().__init__(info, accl, model, input_shape_info, use_model_shape_in)
        self._callback = None
        self._thread = None
        self._stop_event = threading.Event()
        self._pre_model = _PassThroughModel()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._generator_iter = None
        self._worker_exc_log = []

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._callback is not None

    def connect(self, callback):
        sig = inspect.signature(callback)
        if sig.parameters:
            raise RuntimeError(
                "Input `callback` must not have any parameters other than "
                "the implicit self for bound methods"
            )
        self._callback = callback
        if self._thread is not None:
            self.stop()
            self._stop_event = threading.Event()
        model_idx = self._accl.models.index(self._model)
        thread_name = f"Model {model_idx} input function"
        self._thread = threading.Thread(
            target=self._worker,
            args=(callback, self._worker_exc_log),
            name=thread_name,
        )
        self._thread.start()

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            if len(self._worker_exc_log) > 0:
                raise self._worker_exc_log[-1] from None

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def stopped(self):
        return self._stop_event.is_set()

    def _get_frames(self, callback):
        if self._generator_iter is not None:
            frames = advance_iter(self._generator_iter)
        else:
            try:
                cb_result = callback()
            except StopIteration:  # for the case where the app code calls next(iter)
                return None
            if cb_result is None:
                return None
            elif hasattr(cb_result, "__next__"):  # generator or iterator
                self._generator_iter = cb_result
                frames = advance_iter(cb_result)
            else:
                frames = cb_result
        validate_input_callback_result(frames)
        return frames

    @graceful_shutdown
    def _worker(self, callback, worker_exc_log):
        model_idx = self._accl.models.index(self._model)
        while not self._stop_event.is_set():
            try:
                input_frames = self._get_frames(callback)
            except Exception as e:
                logger.critical(
                    f"Model {model_idx} input stream terminated due to an exception related to the callback function: {str(e)}"
                )
                worker_exc_log.append(e)
                self._stop_event.set()
                raise e from None

            if input_frames is None:
                self._stop_event.set()
                break

            try:
                input_frames = self._pre_model.predict_for_mxa(
                    input_frames,
                    self._use_model_shape_in,
                    pre_model=True,
                    output_order=self._info["layers"],
                )
            except Exception as e:
                logger.critical(
                    f"Model {model_idx} input stream terminated due to an exception related to the pre-processing function: {str(e)}"
                )
                worker_exc_log.append(e)
                raise RuntimeError(
                    "Failed to run inference on the pre-processing model\n"
                    "Ensure that the pre-processing model passed to `set_preprocessing_model()` "
                    "and the model in the DFP come from the same model passed to the NeuralCompiler"
                ) from None

            try:
                self._stream_ifmaps(input_frames)
            except Exception as e:
                logger.critical(
                    f"Model {model_idx} input stream terminated due to an exception during streaming inputs: {str(e)}"
                )
                worker_exc_log.append(e)
                raise e from None

            with self._call_counter_lock:
                self._call_counter += 1

            # notify output thread here comes a new ifmap
            with self._model.cond_output:
                self._model.cond_output_flag = True
                self._model.cond_output.notify()

        # done
        logger.debug(f"Model {model_idx} input thread done. Notify output thread to flush out ofmaps")

        # notify output thread to flush out ofmaps left in the driver output queue
        with self._model.cond_flush:
            self._model.cond_flush_flag = True
            self._model.cond_flush.notify()


class AsyncAcclModelOutput(AcclModelOutput):
    def __init__(
        self, info, accl, model, output_shape_info, use_model_shape_out=True
    ):
        super().__init__(
            info, accl, model, output_shape_info, use_model_shape_out
        )
        self._callback = None
        self._thread = None
        self._stop_event = threading.Event()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._worker_exc_log = []

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._callback is not None

    def connect(self, callback):
        self._callback = callback
        if self._thread is not None:
            self.stop()
            self._stop_event = threading.Event()
        self._model_idx = self._accl.models.index(self._model)
        thread_name = f"Model {self._model_idx} output function"
        self._thread = threading.Thread(
            target=self._worker,
            args=(callback, self._model.input, self._model_idx, self._worker_exc_log),
            name=thread_name,
        )
        self._thread.start()

    def wait(self):
        if self._thread is not None:
            self._thread.join()
            if len(self._worker_exc_log) > 0:
                raise self._worker_exc_log[-1] from None

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    def stopped(self):
        return self._stop_event.is_set()

    @graceful_shutdown
    def _worker(self, callback, model_input, model_idx, exc_log):
        def _all_frames_processed():
            with model_input._call_counter_lock:
                all_done = self._call_counter == model_input._call_counter
            return all_done

        outputs = []
        outputs_by_name = {}
        while not self._stop_event.is_set():

            if model_input.stopped() and _all_frames_processed():
                break

            if _all_frames_processed():
                # if we got here, it means input model is not stopped, but haven't sent any new ifmap
                # So wait for it patiently
                with self._model.cond_output:
                    # block at most 0.5 seconds
                    logger.debug(f"Model {model_idx} output thread waiting for input thread to send new ifmap")
                    self._model.cond_output.wait_for(lambda: self._model.cond_output_flag, timeout=0.5)
                    self._model.cond_output_flag = False
                    continue

            success = self._stream_ofmaps(model_idx, outputs, outputs_by_name)
            if not success:
                continue

            try:
                fmaps = self._post_model.predict_for_mxa(
                    outputs_by_name, self._use_model_shape_out
                )
            except Exception as e:
                exc_log.append(e)
                model_input.stop()
                raise RuntimeError(
                    "Failed to run inference on the post-processing model\n"
                    "Ensure that the post-processing model passed to `set_postprocessing_model()` "
                    "and the model in the DFP come from the same model passed to the NeuralCompiler"
                ) from None

            try:
                callback(*fmaps)
            except Exception as e:
                exc_log.append(e)
                logging.critical(
                    f"Model {model_idx} output processing terminated due to an exception in the callback function: {str(e)}"
                )
                model_input.stop()
                raise e from None

            with self._call_counter_lock:
                self._call_counter += 1

        # done
        logger.debug(f"Model {model_idx} output thread done. Waiting for input thread to finish and starts flushing out ofmaps")

        # once the input thread is done, it will set cond_flush_flag as true and notify this thread to flush out ofmaps
        flush_cnt = 0
        with self._model.cond_flush:
            # block at most 1 seconds
            result = self._model.cond_flush.wait_for(lambda: self._model.cond_flush_flag, timeout=1)
            if not result:
                logger.warning(f"Model {model_idx} timeout: output thread did not receive flush signal")
            self._model.cond_flush_flag = False

            # flush out ofmaps left in the driver output queue
            logger.debug(f"Model {model_idx} starts to flush out ofmaps")
            while not _all_frames_processed():

                success = self._stream_ofmaps(model_idx, outputs, outputs_by_name)
                if not success:
                    continue

                flush_cnt += 1
                with self._call_counter_lock:
                    self._call_counter += 1

        logger.debug(f"Model {model_idx} output thread flushed out {flush_cnt} ofmaps")


class MultiStreamAsyncAccl(Accl):
    """
    This class provides a multi-stream version of the AsyncAccl API. This allows multiple input+output callbacks
    to be associated with a single model.

    Parameters
    ----------
    dfp : bytes or str
        Path to the DFP file generated by the NeuralCompiler, or a byte array
        representing the DFP content.

    device_ids : list of int, optional
        List of MemryX device IDs to be used for executing the DFP.
        Default is [0].
    
    stream_workers : int, optional
        Number of worker threads to use for thread pooling multiple stream threads
        If -1, the number of workers is set to (CPU cores - 1), with a minimum of 2.
        This allows for efficient asynchronous data flow without overwhelming the system.
        Default is -1.

    use_model_shape : tuple of bool, optional
        Tuple in the form (input_shape, output_shape). Specifies whether to enforce
        the original model input/output shapes (True) or use MXA runtime shapes (False).
        Default is (True, True).

    local_mode : bool, optional
        If True, executes the DFP in local mode, which can improve performance for
        single-process use. Incompatible with multi-DFP and multi-process use.
        Default is False.

    scheduler_options : SchedulerOptions, optional
        Scheduler configuration corresponding to the SchedulerOptions struct:
        
        - frame_limit : int  
          Number of frames to process before swapping out. Default 20.
        - time_limit : int  
          Maximum idle time (in milliseconds) before swapping out. Default 250.
        - stop_on_empty : bool  
          Whether to immediately swap if the input queue is empty. Default False.
        - ifmap_queue_size : int  
          Size of the shared input feature map queue. Default 16.
        - ofmap_queue_size : int  
          Size of the per-client output feature map queues. Default 21.

    client_options : ClientOptions, optional
        Client runtime behavior configuration, corresponding to the ClientOptions struct:

        - smoothing : bool  
          Whether to enable FPS smoothing. Default False.
        - fps_target : float  
          Target frames per second. A delay of 1 / fps_target seconds is enforced 
          between frames. Default 0.

    manager_addr : str, optional
        Path to the mxa-manager socket. Needed in Docker or managed environments.
        Default is "/run/mxa_manager/".

    manager_port : int, optional
        Port for mxa-manager connection (used primarily in containerized deployments).
        Default is 10000.

    ignore_manager : bool, optional
        If True, bypasses the manager and forces local mode. May cause crashes if
        multiple containers/processes attempt to use the same device.
        Default is False.
    
    Warnings
    --------
    Setting `ignore_manager` to True disables coordination with other processes and may result 
    in device conflicts if multiple clients or containers access the same device concurrently. 
    Use with caution.



    Examples
    --------

    .. code-block:: python

        from tensorflow import keras
        import numpy as np
        from memryx import NeuralCompiler, MultiStreamAsyncAccl

        class Application:
            def __init__(self):
                self.streams = []
                self.streams_idx = []
                self.outputs = []
                for i in range(2):
                    self.streams.append(
                        [
                            np.random.rand(224, 224, 3).astype(np.float32)
                            for _ in range(10)
                        ]
                    )
                    self.streams_idx.append(0)
                    self.outputs.append([])

            # Define the callback that will return model input data
            def data_source(self, stream_idx):
                # Generate inputs based on stream_idx
                if self.streams_idx[stream_idx] == len(self.streams[stream_idx]):
                    return None
                self.streams_idx[stream_idx] += 1
                return self.streams[stream_idx][self.streams_idx[stream_idx] - 1]

            # Define the callback that will process the outputs of the model
            def output_processor(self, stream_idx, *outputs):
                logits = np.squeeze(outputs[0], 0)
                preds = keras.applications.mobilenet.decode_predictions(logits)
                # Route outputs based on stream_idx
                self.outputs[stream_idx].append(preds)

        # Compile a MobileNet model for testing.
        # Typically, comilation need to be done one time only.
        model = keras.applications.MobileNet()
        nc = NeuralCompiler(models=model, verbose=1)
        dfp = nc.run()

        # Accelerate using the MemryX hardware
        app = Application()
        accl = MultiStreamAsyncAccl(dfp)

        # Starts asynchronous execution of input output callback pair associated with 2 streams
        accl.connect_streams(app.data_source, app.output_processor, 2)

        # Wait for the accelerator to finish execution
        accl.wait()

        # Explicitly free up hardware
        accl.shutdown()

    """

    def __init__(
        self,
        dfp: Union[str, Path, bytes, Dfp],
        device_ids: Union[List[int], int] = [0],
        stream_workers: int = -1,
        use_model_shape: Tuple[bool, bool] = (True, True),
        local_mode: bool = False,
        scheduler_options: SchedulerOptions =  SchedulerOptions(),
        client_options: ClientOptions = ClientOptions(),
        manager_addr: str = "/run/mxa_manager/",
        manager_port: int = 10000,
        ignore_manager: bool = False,
    ):
        self.use_model_shape = use_model_shape
        if stream_workers == -1:
            cpu_count = os.cpu_count()
            cpu_count = 0 if cpu_count is None else cpu_count
            stream_workers = cpu_count - 1
        stream_workers = max(2, stream_workers)
        self._input_task_pool = InputTaskPool(stream_workers, self.use_model_shape[0])
        super().__init__(dfp, device_ids, use_model_shape, local_mode, scheduler_options, client_options, manager_addr, manager_port, ignore_manager)
        self._input_tasks = {}

    def _create_models(self):
        self._models: list[MultiStreamAsyncAcclModel] = []
        self._stream_idx = []
        for i, m in enumerate(self._model_idx_to_in_info):
            self._models.append(
                MultiStreamAsyncAcclModel(
                    self._model_idx_to_in_info[m],
                    self._model_idx_to_out_info[m],
                    self,
                    m,
                    str(i),
                    self.input_shape_info,
                    self.output_shape_info,
                    use_model_shape=self._use_model_shape,
                )
            )
            self._stream_idx.append(0)

    def connect_streams(
        self, input_callback, output_callback, stream_count, model_idx=0
    ):
        """
        Registers and starts execution of a pair of input and output callback functions
        that process a specified number of data streams as defined by `stream_count` for a given model.

        Parameters
        ----------
        input_callback : callable
            A function or bound method responsible for supplying input data to the model
            identified by `model_idx`. It must accept exactly one argument:

            - stream_idx : int  
              The index of the input stream, ranging from 0 to `stream_count` - 1,
              used by the application to select the appropriate data source.

        output_callback : callable
            A function or bound method invoked with the output feature maps generated by the model.
            It must accept at least two arguments:

            - stream_idx : int 
              The index of the output stream (same as input).
            - \*ofmaps : tuple or unpacked 
              The output feature maps. The function can use a variadic signature (\*ofmaps) or 
              named parameters (e.g., fmap0, fmap1, ...) depending on how many feature maps 
              the model produces.

        stream_count : int
            The number of independent input feature map sources (streams) that the model will process.

        model_idx : int, optional
            The index of the model to associate with this input-output callback pair.
            Each stream for the specified model will be assigned a stream_idx in the
            range [0, `stream_count` - 1].
            Default is 0.
        """
        if not isinstance(stream_count, int):
            raise TypeError("stream_count must be an integer")
        if stream_count <= 0:
            raise ValueError("stream_count must be positive")
        sig = inspect.signature(input_callback)
        if len(sig.parameters) != 1:
            raise TypeError(
                "input_callback must have exactly 1 parameter (stream index) "
                "other than the implicit self for bound methods"
            )
        sig = inspect.signature(output_callback)
        if len(sig.parameters) < 2:
            raise TypeError(
                "output_callback must have at least 2 parameters (stream index, *fmaps) "
                "other than the implicit self for bound methods"
            )
        self._ensure_valid_model_idx(model_idx)
        for i in range(stream_count):
            stream = Stream(
                model_idx, self._stream_idx[model_idx], input_callback, output_callback
            )
            task = InputTask(self._models[model_idx], stream)
            task_key = self._get_task_key(
                self._stream_idx[model_idx], input_callback, output_callback
            )
            self._input_tasks[task_key] = task
            self._input_task_pool.add_task(task)
            self._stream_idx[model_idx] += 1
        try:
            self._models[model_idx]._connect_stream(stream_count)
        except Exception as e:
            self.stop()
            raise e from None

    def stop(self):
        """
        Sends a signal to stop each of the models running on the accelerator.
        This call blocks until each of the models stops and cleans up its
        resources.
        """
        self._input_task_pool.stop()
        for model in self._models:
            model.stop()

    def wait(self):
        """
        Blocks the application thread until the accelerator finishes executing all models.

        Raises
        ------
        RuntimeError: If the any of the model's inputs/outputs are left unconnected
        """
        try:
            for model in self._models:
                model.wait()
        except KeyboardInterrupt:
            for model in self._models:
                model.stop()
        self._input_task_pool.stop()

    def _get_task_key(self, index, input_callback, output_callback):
        return (index, input_callback, output_callback)


class InputTask:
    def __init__(self, model, stream):
        self.model = model
        self.stream = stream

    def __str__(self):
        return f"Model {self.model.name()} stream {self.stream.index}"


class InputTaskPool:
    def __init__(self, count, use_model_shape_in):
        self._count = count
        self._use_model_shape_in = use_model_shape_in
        self._task_queue = queue.Queue()
        self._tasks = set()
        self._workers = []
        self._lock = threading.Lock()
        for i in range(count):
            worker_name = f"InputWorker-{i}"
            self._workers.append(
                threading.Thread(
                    target=self._worker_target, args=(), daemon=True, name=worker_name
                )
            )
        self._stop_event = threading.Event()
        for worker in self._workers:
            worker.start()

    def _worker_target(self):
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                logger.debug("Input worker get task timed out")
                continue
            logger.debug(f"Input worker dequeued {task}")
            if task not in self._tasks:
                continue
            logger.debug(f"Input worker executing {task} callback")
            input_frames = self._execute_task(task)
            logger.debug(f"Input worker received frames from {task} callback")
            self._enqueue_model_input_frames(task, input_frames)
            finished = input_frames is None
            if finished:
                logger.debug(f"{task} ended")
                self._tasks.remove(task)
                continue
            self._task_queue.put(task)

    def _execute_task(self, task):
        stream_idx = task.stream.index

        try:
            input_frames = self._get_frames(task.stream.input_callback, stream_idx)
        except Exception as e:
            logger.critical(
                f"Model {task.model.name()} input stream {stream_idx} terminated due to an exception related to the callback function: "
            )
            logger.critical("AssertionError") if isinstance(
                e, AssertionError
            ) else logger.critical(str(e))
            raise e from None

        if input_frames is None:
            return input_frames

        mpu_input_order = task.model.input._info["layers"]

        self._lock.acquire()
        try:
            input_frames = task.model.input._pre_model.predict_for_mxa(
                input_frames,
                self._use_model_shape_in,
                pre_model=True,
                output_order=mpu_input_order,
            )
        except Exception:
            raise RuntimeError(
                "Failed to run inference on the pre-processing model\n"
                "Ensure that the pre-processing model passed to `set_preprocessing_model()` "
                "and the model in the DFP come from the same model passed to the NeuralCompiler"
            ) from None
        self._lock.release()

        return input_frames

    def _enqueue_model_input_frames(self, task, input_frames):
        while not self._stop_event.is_set():
            try:
                task.model.input.put(input_frames, task.stream, timeout=0.5)
            except queue.Full:
                logger.debug(f"Model {task.model.name()} input queue full")
            else:
                logger.debug(f"Model {task.model.name()} input queue got frames")
                return

    def _get_frames(self, callback, *cb_args):
        try:
            cb_result = callback(*cb_args)
        except StopIteration:
            return None
        if cb_result is None:
            return None
        if type(cb_result).__name__ == "generator":
            raise TypeError("input_callback must not return a generator")
        elif hasattr(cb_result, "__next__"):
            frames = advance_iter(cb_result)
        else:
            frames = cb_result
        validate_input_callback_result(frames)
        return frames

    def add_task(self, task):
        self._task_queue.put(task)
        self._tasks.add(task)
        logger.debug(f"Enqueued {task}")

    def remove_task(self, task):
        if not self.has_task(task):
            raise ValueError("task not found")
        self._tasks.remove(task)

    def has_task(self, task):
        return task in self._tasks

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        logger.debug("Input worker pool stop signal sent")
        for worker in self._workers:
            worker.join()

    def stopped(self):
        return self._stop_event.is_set()


class MultiStreamAsyncAcclModel(AcclModel):
    def __init__(
        self,
        input_info,
        output_info,
        accl,
        model_idx,
        name,
        input_shape_info,
        output_shape_info,
        use_model_shape=(True,True),
    ):
        super().__init__(
            input_info, output_info, accl, input_shape_info, output_shape_info
        )
        self._name = name
        self.model_idx = model_idx
        self.input = MultiStreamAsyncAcclModelInput(
            input_info, accl, self, input_shape_info, use_model_shape[0]
        )
        self.output = MultiStreamAsyncAcclModelOutput(
            output_info, accl, self, output_shape_info, use_model_shape[1]
        )

    def _connect_stream(self, stream_count):
        self.output.connect(stream_count)
        self.input.connect(stream_count)

    def wait(self):
        self.input.wait()
        self.output.wait()
        logger.debug(f"Model {self._name} wait ended")

    def stop(self):
        self.input.stop()
        self.output.stop()
        logger.debug(f"Model {self._name} stopped")

    def name(self):
        return self._name


class Stream:
    def __init__(self, model_idx, index, input_callback, output_callback):
        self.model_idx = model_idx
        self.index = index
        self.input_callback = input_callback
        self.output_callback = output_callback

    def __str__(self):
        return f"Stream {self.index}"


class MultiStreamAsyncAcclModelInput(AcclModelInput):
    def __init__(
        self, info, accl, model, input_shape_info, use_model_shape_in=True
    ):
        super().__init__(
            info,
            accl,
            model,
            input_shape_info,
            use_model_shape_in=use_model_shape_in,
        )
        self._connected = False
        self._stop_event = threading.Event()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._merge_queue = queue.Queue(
            maxsize=10
        )  # TODO: consider a max-size based on frame memory consumption and expose to user?
        self._output_stream_queue = queue.Queue()
        self._pre_model = _PassThroughModel()
        self._thread = threading.Thread(
            target=self._worker,
            args=(self._merge_queue, self._output_stream_queue),
            name=f"Model {model.name()} InputWorker",
        )
        self._stream_log = deque(
            [], maxlen=1000
        )  # for making assertions stream routing in tests
        self._stream_count = 0

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._connected

    def connect(self, stream_count):
        """
        Set an input callback
        """
        self._connected = True
        self._stream_count += stream_count
        if not self._thread.is_alive():
            self._thread.start()

    def put(self, input_frames, stream, timeout):
        self._merge_queue.put((input_frames, stream), timeout=timeout)

    def wait(self):
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} input wait ended")

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        logger.debug(f"Model {self._model.name()} input stop signal sent")
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} input stopped")

    def stopped(self):
        return self._stop_event.is_set()

    @graceful_shutdown
    def _worker(self, merge_queue, output_stream_queue):
        done_count = 0
        while not self._stop_event.is_set():
            try:
                input_frames, stream = merge_queue.get(timeout=0.5)
            except queue.Empty:
                logger.debug(f"Model {self._model.name()} get frames timed out")
                continue
            if input_frames is None:
                done_count += 1
                logger.debug(
                    f"Model {self._model.name()} finished stream count: {done_count}"
                )
                if done_count == self._stream_count:
                    self._stop_event.set()
                    logger.debug(f"Model {self._model.name()} finished all streams")
                    break
                continue

            self._stream_log.append(stream.index)
            output_stream_queue.put(stream)

            self._stream_ifmaps(input_frames)
            logger.debug(f"Model {self._model.model_idx} stream inputs successful")

            with self._call_counter_lock:
                self._call_counter += 1

            # notify output thread here comes a new ifmap
            with self._model.cond_output:
                self._model.cond_output_flag = True
                self._model.cond_output.notify()
                
        # done
        logger.debug(f"Model {self._model.model_idx} input thread done. Notify output thread to flush out ofmaps")

        # notify output thread to flush out ofmaps left in the driver output queue
        with self._model.cond_flush:
            self._model.cond_flush_flag = True
            self._model.cond_flush.notify()
            

class MultiStreamAsyncAcclModelOutput(AcclModelOutput):
    def __init__(
        self, info, accl, model, output_shape_info, use_model_shape_out=True
    ):
        super().__init__(
            info, accl, model, output_shape_info, use_model_shape_out
        )
        self._connected = False
        self._stop_event = threading.Event()
        self._post_model = _PassThroughModel()
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._worker,
            args=(model,),
            name=f"Model {model.name()} OutputWorker",
        )
        self._stream_log = deque([], maxlen=1000)
        self._stream_count = 0
        self._use_model_shape_out = use_model_shape_out

    def connected(self):
        """
        Return whether a callback was set for execution
        """
        return self._connected

    def connect(self, stream_count):
        self._connected = True
        self._stream_count += stream_count
        if not self._thread.is_alive():
            self._thread.start()

    def wait(self):
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} output wait ended")

    def stop(self):
        if self.stopped():
            return
        self._stop_event.set()
        logger.debug(f"Model {self._model.name()} output stop signal sent")
        if self._thread.is_alive():
            self._thread.join()
        logger.debug(f"Model {self._model.name()} output stopped")

    def stopped(self):
        return self._stop_event.is_set()

    def _all_frames_processed(self):
        with self._model.input._call_counter_lock:
            all_done = self._call_counter == self._model.input._call_counter
        return all_done

    @graceful_shutdown
    def _worker(self, model):
        outputs = []
        outputs_by_name = {}

        while not self._stop_event.is_set():
            
            if model.input.stopped() and self._all_frames_processed():
                break
            
            if self._all_frames_processed():
                # if we got here, it means input model is not stopped, but haven't sent any new ifmap
                # So wait for it patiently
                with model.cond_output:
                    # block at most 0.5 seconds
                    logger.debug(f"Model {model.model_idx} output thread waiting for input thread to send new ifmap")
                    model.cond_output.wait_for(lambda: model.cond_output_flag, timeout=0.5)
                    model.cond_output_flag = False
                    continue
                
            success = self._stream_ofmaps(model.model_idx, outputs, outputs_by_name)
            if not success:
                continue

            stream = self._model.input._output_stream_queue.get()
            self._stream_log.append(stream.index)

            try:
                fmaps = self._post_model.predict_for_mxa(
                    outputs_by_name, self._use_model_shape_out
                )
            except Exception:
                raise RuntimeError(
                    "Failed to run inference on the post-processing model\n"
                    "Ensure that the post-processing model passed to `set_postprocessing_model()` "
                    "and the model in the DFP come from the same model passed to the NeuralCompiler"
                ) from None

            try:
                stream.output_callback(stream.index, *fmaps)
            except Exception as e:
                logger.critical(
                    f"{stream} output processing terminated due to an exception in the callback function:"
                )
                self._model.input.stop()
                raise e from None

            with self._call_counter_lock:
                self._call_counter += 1

        # flush out ofmaps left in the driver output queue
        # done
        logger.debug(f"Model {model.model_idx} output thread done. Waiting for input thread to finish and starts flushing out ofmaps")

        # once the input thread is done, it will set cond_flush_flag as true and notify this thread to flush out ofmaps
        flush_cnt = 0
        with self._model.cond_flush:
            # block at most 1 seconds
            result = self._model.cond_flush.wait_for(lambda: self._model.cond_flush_flag, timeout=1)
            if not result:
                logger.warning(f"Model {model.model_idx} timeout: output thread did not receive flush signal")
            self._model.cond_flush_flag = False

            # flush out ofmaps left in the driver output queue
            logger.debug(f"Model {model.model_idx} starts to flush out ofmaps")
            while not self._all_frames_processed():

                success = self._stream_ofmaps(model.model_idx, outputs, outputs_by_name)
                if not success:
                    continue

                flush_cnt += 1
                with self._call_counter_lock:
                    self._call_counter += 1

        logger.debug(f"Model {model.model_idx} output thread flushed out {flush_cnt} ofmaps")
