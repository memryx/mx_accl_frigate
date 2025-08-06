import sys
from collections import defaultdict, deque
from pathlib import Path
import logging
import numpy as np

try:
    import memryx
except ImportError:
    mix_home = Path(__file__).parent.parent.resolve()
    sys.path.append(str(mix_home))
    import memryx

from memryx.runtime.powermode import get_power_tuple
from memryx.runtime.dfp import Dfp

logger = logging.getLogger(__name__)

CHIP_NAME_TO_GEN = {"Cascade": 3, "Cascade+": 3.1}


class DFPRunner:
    def __init__(self, dfp_obj: Dfp, manager_addr: str,
                 base_port: int, local_mode: bool, device_ids_to_use: list[int], ignore_manager: bool):
        """
        DFPRunner class is used to manage the DFP execution, either through local mode or shared mode.
        In local mode, it creates a single client and locally manages driver calls like memx_open, etc.
        """
        self._dfp_obj = dfp_obj
        self._manager_addr = manager_addr
        self._base_port = base_port
        self._local_mode = local_mode
        self._device_ids_to_use = device_ids_to_use
        self._ignore_manager = ignore_manager
        self._clients = []
        self._local_open_contexts = []
        self._local_locked_device_ids = []

        self._inport_info = {}
        self._outport_info = {}
        
        self.in_cnt = 0 # for debug
        self.out_cnt = 0 # for debug

    def _clear_local(self):

        if not self._ignore_manager:
            # runner is already cleared
            if len(self._clients) == 0:
                return
            
        # memx_close any open contexts
        for context_id in self._local_open_contexts:
            memryx.mxa.close(context_id)

        if self._ignore_manager:
            for i in self._local_locked_device_ids:
                if memryx.mxa.unlock(i) != 0:
                    logger.error(f"Error unlocking device id: {i}")
        else:
            # unlock any that were locked by us
            for i in self._local_locked_device_ids:
                self._clients[0].local_unlock(i)
            self._clients.clear()

        self._local_open_contexts.clear()

    def init_local(self) -> bool:

        logger.debug(f" Initializing local mode")

        if len(self._device_ids_to_use) > 1:
            raise RuntimeError(
                "DFPRunner only supports running one device in local mode. "
                "Please switch to shared mode to run multiple devices."
            )

        # ==============================================
        # NOTE: 
        # - mxa   returns "failure" (True on failure)
        # - mxapi returns "success" (True on success)
        #
        # Be careful and don't be confused!
        # ==============================================

        ############### Lock device using mxa manager ###############

        if not self._ignore_manager:
            # create client
            client = memryx.mxapi.Client()

            # connect to the mxrpc manager
            if client.init_connection(self._manager_addr, self._base_port) == False:
                logger.error(
                    f"[DFPRunner] Error in client->init_conenction local mode for device: FIXME")
                return False

            self._clients.append(client)

        # for each device id in the vector, do try_local_lock and if any of them
        # fail, return false (after first unlocking any that were locked by us)
        # use a temporary vector to store the device ids that we locked
        device_id = self._device_ids_to_use[0]

        if self._ignore_manager:
            if memryx.mxa.trylock(device_id) != 0:
                logger.error(
                    f" Error in memx_trylock for device id: {device_id}")
                self._clear_local()
                return False
            else:
                self._local_locked_device_ids.append(device_id)
        else:
            # lock device
            if client.try_local_lock(device_id) == False:
                logger.error(
                    f" Error in client->try_local_lock for device id: {device_id}")
                self._clear_local()
                return False

            else:
                self._local_locked_device_ids.append(device_id)

        ############### open the memx_open context for the device id (1:1 mapping) ###############
        driver_context_id = device_id

        # open the driver context
        if memryx.mxa.open(driver_context_id, device_id, 3.1) == True:
            logger.error(
                f" Error in memx_open for device id: {device_id}")
            self._clear_local()
            return False

        else:
            logger.debug(
                f" Successfully opened context for device id: {device_id}")
            self._local_open_contexts.append(driver_context_id)


        ############### set_power_mode for each opened device ###############

        dfp_mpus = self._dfp_obj.num_chips
        hw_mpus = memryx.mxa.get_total_chip_count(device_id)

        if hw_mpus == 4 and dfp_mpus == 2:
            memryx.mxa.config_mpu_group(device_id, 1)

            powercfg = get_power_tuple()
            memryx.mxa.set_mpu_frequency(device_id, 0, powercfg[2])
            memryx.mxa.set_mpu_frequency(device_id, 1, powercfg[2])
            memryx.mxa.set_mpu_voltage(device_id, powercfg[3])

        elif hw_mpus == 4 and dfp_mpus == 4:
            memryx.mxa.config_mpu_group(device_id, 0)

            powercfg = get_power_tuple()
            memryx.mxa.set_mpu_frequency(device_id, 0, powercfg[0])
            memryx.mxa.set_mpu_frequency(device_id, 1, powercfg[0])
            memryx.mxa.set_mpu_frequency(device_id, 2, powercfg[0])
            memryx.mxa.set_mpu_frequency(device_id, 3, powercfg[0])
            memryx.mxa.set_mpu_voltage(device_id, powercfg[1])

        elif hw_mpus == 8 and dfp_mpus == 8:
            memryx.mxa.config_mpu_group(device_id, 5)

        elif hw_mpus == 2 and dfp_mpus == 2:
            memryx.mxa.config_mpu_group(device_id, 4)

            powercfg = get_power_tuple()
            memryx.mxa.set_mpu_frequency(device_id, 0, powercfg[2])
            memryx.mxa.set_mpu_frequency(device_id, 1, powercfg[2])
            memryx.mxa.set_mpu_voltage(device_id, powercfg[3])
        else:
            logger.error(
                f"Input DFP was compiled for {dfp_mpus} chips, but the connected accelerator has {hw_mpus} chips")
            self._clear_local()
            return False

        ############### Download the DFP to each open device ###############
        if memryx.mxa.download_buffer(driver_context_id, self._dfp_obj._dfp_bytes.getvalue()):
            logger.error("Error loading DFP, try resetting the board")
            self._clear_local()
            return False

        ############### Set stream enable for each open context ###############
        if memryx.mxa.set_stream_enable(driver_context_id, 0):
            logger.error(
                "Error enabling input/output, try resetting the board")
            self._clear_local()
            return False

        return True  # success

    def init_shared(self, sched_options_params: tuple, client_options_params: tuple) -> bool:
        logger.debug(f" Initializing shared mode")

        # setup _inport_info and _outport_info for use in shared-mode stream_ifmap and stream_ofmap later
        self._inport_info = self._dfp_obj.input_ports
        for p in self._inport_info.values():
            # set numerical format from strings
            if p['packing_format'] == "bf16":
                p['format'] = int(4)
            elif p['packing_format'] == "gbfloat80" or p['packing_format'] == "gbf80":
                p['format'] = int(0)
            elif p['packing_format'] == "gbf80_row" or p['packing_format'] == "gbfloat80_row":
                p['format'] = int(6)
            elif p['packing_format'] == "fp32" or p['packing_format'] == "float32":
                p['format'] = int(5)
            else:
                logger.error(
                    f"Unsupported packing format {p['packing_format']} for input port {p['name']}")
                return False

            # pre-calculate the tensor_size (just muliply the shape dimensions)
            fmap_size = int(np.prod(p['shape']))
            dim_h = int(p['shape'][0])
            dim_w = int(p['shape'][1])
            dim_z = int(p['shape'][2])
            dim_c = int(p['shape'][3])
            p['tensor_size'] = fmap_size

            # pre-calculate the total size in bytes based on format
            if p['format'] == 5:
                p['fmt_size'] = p['tensor_size'] * 4
            elif p['format'] == 4:
                p['fmt_size'] = p['tensor_size'] * 2
                # pad to 4-bytes
                if p['tensor_size'] % 2 != 0:
                    p['fmt_size'] += 2
            elif p['format'] == 0:
                any_remainder_chs = ((dim_c % 8) != 0)
                num_xyz_pixels = int( fmap_size // dim_c )
                num_gbf_words = int(int( dim_c // 8 ) + ( 1 if any_remainder_chs else 0 ))
                p['fmt_size'] = int(num_xyz_pixels * num_gbf_words * 10)
                # pad to 4-bytes
                if p['fmt_size'] % 4 != 0:
                    p['fmt_size'] += 4 - (p['fmt_size'] % 4)
            elif p['format'] == 6:
                any_remainder_chs = ((dim_c % 8) != 0)
                num_gbf_words = int(int( dim_c // 8 ) + ( 1 if any_remainder_chs else 0 ))
                row_size_rounded = int( dim_w * dim_z * num_gbf_words * 10 )
                if row_size_rounded % 4 != 0:
                    row_size_rounded += 4 - (row_size_rounded % 4)
                p['fmt_size'] = int(row_size_rounded * dim_h)

            # make an np.array dtype=np.uint8 with the size of fmt_size
            p['buffer'] = np.zeros(p['fmt_size']+4, dtype=np.uint8)  # +4 for paranoia

        # do the same for output ports, but also need to consider HPOC
        self._outport_info = self._dfp_obj.output_ports
        for p in self._outport_info.values():
            # set numerical format from strings
            if p['packing_format'] == "bf16":
                p['format'] = int(4)
            elif p['packing_format'] == "gbfloat80" or p['packing_format'] == "gbf80":
                p['format'] = int(0)
            elif p['packing_format'] == "gbf80_row" or p['packing_format'] == "gbfloat80_row":
                p['format'] = int(6)
            elif p['packing_format'] == "fp32" or p['packing_format'] == "float32":
                p['format'] = int(5)
            else:
                logger.error(
                    f"Unsupported packing format {p['packing_format']} for output port {p['name']}")
                return False

                

            # pre-calculate the tensor_size (just muliply the shape dimensions)
            fmap_size = int(np.prod(p['shape']))
            dim_h = int(p['shape'][0])
            dim_w = int(p['shape'][1])
            dim_z = int(p['shape'][2])
            dim_c = int(p['shape'][3])
            
            # set HPOC
            true_dim_c = int(p['hpoc_fm_shape'][3]) if p['hpoc_enabled'] else dim_c
            if p['hpoc_enabled']:
                p['hpoc_extra_size'] = len(p['hpoc_list'])
                p['hpoc_indexes'] = np.array(p['hpoc_list'], dtype=np.int32)
            else:
                p['hpoc_extra_size'] = 0
                p['hpoc_indexes'] = np.array([0], dtype=np.int32)
            p['tensor_size'] = fmap_size

            # pre-calculate the total size in bytes based on format, and also consider HPOC
            if p['format'] == 5:
                p['fmt_size'] = p['tensor_size'] * 4
            elif p['format'] == 4:
                p['fmt_size'] = p['tensor_size'] * 2
                # pad to 4-bytes
                if p['tensor_size'] % 2 != 0:
                    p['fmt_size'] += 2
            elif p['format'] == 0:
                any_remainder_chs = ((true_dim_c % 8) != 0)
                num_xyz_pixels = int( fmap_size // dim_c ) # de-HPOCd "real" shape
                num_gbf_words = int(int( true_dim_c // 8 ) + ( 1 if any_remainder_chs else 0 ))
                p['fmt_size'] = int(num_xyz_pixels * num_gbf_words * 10)
                # pad to 4-bytes
                if p['fmt_size'] % 4 != 0:
                    p['fmt_size'] += 4 - (p['fmt_size'] % 4)
            elif p['format'] == 6:
                any_remainder_chs = ((true_dim_c % 8) != 0)
                num_gbf_words = int(int( true_dim_c // 8 ) + ( 1 if any_remainder_chs else 0 ))
                row_size_rounded = int( dim_w * dim_z * num_gbf_words * 10 )
                # pad to 4-bytes
                if row_size_rounded % 4 != 0:
                    row_size_rounded += 4 - (row_size_rounded % 4)
                p['fmt_size'] = int(row_size_rounded * dim_h)
            
            # make an np.array dtype=np.uint8 with the size of fmt_size
            p['buffer'] = np.zeros(p['fmt_size']+4, dtype=np.uint8) # +4 for paranoia


        # init client for each model within DFP
        for model_id in range(len(self._dfp_obj.models)):

            # create client
            client = memryx.mxapi.Client()

            # connect to the mxrpc manager
            if client.init_connection(self._manager_addr, self._base_port) == False:
                logger.error(f" Error connecting to mxrpc manager")
                return False

            # Call the connect_dfp method
            sched_options = memryx.mxapi.SchedulerOptions(
                *sched_options_params)
            client_options = memryx.mxapi.ClientOptions(*client_options_params)

            # connect / register DFP to mxrpc manager
            success = client.connect_dfp(
                self._dfp_obj._dfp_bytes.getvalue(),
                model_id,
                sched_options,
                client_options,
                self._device_ids_to_use,
            )

            if not success:
                logger.error(f" Client connect dfp failed")
                return False

            self._clients.append(client)

        return True  # success

    def stream_ifmap(self, model_idx: int, port: int, ifmap: np.ndarray) -> bool:
        """ 
        Stream input data to the device.
        
        local mode: 
            Different models share the same driver context to communicate with the device.

            Schematic diagram:
                    DFP
                +-----------+ 
                |  Model 0  | ---┐   
                |           |    ---> driver_context ----> device
                |  Model 1  | ---┘
                +-----------+

        shared mode:
            Each model is attached to different client, which in turn share the same driver context to communicate with the device.
            
                    DFP
                +-----------+ 
                |  Model 0  | --- client 0 ---┐
                |           |                  ---> driver_context ----> device
                |  Model 1  | --- client 1 ---┘
                +-----------+
        """

        if self._local_mode:

            driver_context_id = self._device_ids_to_use[0]
                
            err = memryx.mxa.stream_ifmap(driver_context_id, port, ifmap)
            if err:
                logger.debug(
                    f"Failed to send ifmap for driver_context_id {driver_context_id}. Error code: {err}")
                return False
        else:
            # ============================================
            # NOTE:
            # No need to iterate over all driver_context_id values here.
            # MX manager handles that internally. Providing the client is sufficient.
            # ============================================

            client = self._clients[model_idx]

            p = self._inport_info[port]
            success = memryx.mxapi.stream_ifmap(
                port, ifmap, client,
                p['fmt_size'],
                p['tensor_size'],
                p['format'],
                p['shape'][0],  # dim_h
                p['shape'][1],  # dim_w
                p['shape'][2],  # dim_z
                p['shape'][3],  # dim_c
                p['buffer']     # temp buffer for format conversion
            )
            
            if not success:
                logger.debug(
                    f"Failed to send ofmap for client {client.my_client_id}.")
                return False

        return True  # success

    def stream_ofmap(self, model_idx: int, port: int, ofmap: np.ndarray, timeout=500) -> bool:
        """ 
        Stream output data from the device.
        
        local mode: 
            Different models share the same driver context to communicate with the device.

            Schematic diagram:
                    DFP
                +-----------+ 
                |  Model 0  | ---┐
                |           |     ---> driver_context ----> device
                |  Model 1  | ---┘
                +-----------+

        shared mode:
            Each model is attached to different client, which in turn share the same driver context to communicate with the device.
            
                    DFP
                +-----------+ 
                |  Model 0  | --- client 0 ---┐
                |           |                  ---> driver_context ----> device
                |  Model 1  | --- client 1 ---┘
                +-----------+
        """
        
        # ============================================
        # NOTE: mxa.stream_ofmap returns "error" (non-zero on failure),
        #       while mxapi.stream_ofmap returns "success" (True on success)
        # ============================================
        if self._local_mode:
            
            driver_context_id = self._device_ids_to_use[0]
            err = memryx.mxa.stream_ofmap(driver_context_id, port, ofmap, timeout)

            if err:
                logger.debug(
                    f"Failed to recv ofmap for driver_context_id {driver_context_id}. Error code: {err}")
                return False
        else:
            # ============================================
            # NOTE:
            # No need to iterate over all driver_context_id values here.
            # MX manager handles that internally. Providing the client is sufficient.
            # ============================================
            
            client = self._clients[model_idx]

            p = self._outport_info[port]
            success = memryx.mxapi.stream_ofmap(
                port, ofmap, client,
                p['fmt_size'],
                p['tensor_size'],
                p['format'],
                p['shape'][0],        # dim_h
                p['shape'][1],        # dim_w
                p['shape'][2],        # dim_z
                p['shape'][3],        # dim_c
                p['hpoc_enabled'],    # hpoc_enabled
                p['hpoc_extra_size'], # hpoc_extra_size
                p['hpoc_indexes'],    # hpoc_indexes
                p['buffer']           # temp buffer for format conversion
            )

            if not success:
                logger.debug(
                    f"Failed to recv ofmap for client {client.my_client_id}.")
                return False

        return True  # success

    def shutdown(self):
        """
        Shutdown the DFP runner and all clients.
        """
        logger.debug(f" Shutting down DFP runner")

        self._dfp_obj = None

        if self._local_mode:
            self._clear_local()
        else:
            for client in self._clients:
                client.end_connection()
            self._clients.clear()
