from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import os
import io

class Model(ABC):
    def __init__(self, loaded_model, model_type):
        self.loaded_model = loaded_model
        self.model_type = model_type
        self._input_names = None
        self._input_shapes = None
        self._output_names = None

    @abstractmethod
    def _minimal_init(self, loaded_model, model_type):
        # used for inference/prediction - accelerator API
        self.loaded_model = loaded_model
        self.type = model_type
        self._input_names = None
        self._input_shapes = None
        self._input_shapes_by_name = None
        self._output_names = None

    @abstractmethod
    def predict_for_mxa(self, inputs, use_model_shape, pre_model=False, output_order=None):
        pass

    def input_names(self):
        return self._input_names

    def input_shapes(self):
        return self._input_shapes

    def output_names(self):
        return self._output_names

class KerasModel(Model):
    def _minimal_init(self, loaded_model, model_type):
        """Initialize Keras model attributes for inference."""
        super()._minimal_init(loaded_model, model_type)
        self._input_names, self._input_shapes, self._input_shapes_by_name = [], [], {}
        for tensor in self.loaded_model.inputs:
            shape = list(tensor.shape)
            if shape[0] is None:
                shape = [1] + shape[1:]  # Replace batch dimension with 1
            self._input_names.append(tensor.node.layer.name)
            self._input_shapes.append(shape)
            self._input_shapes_by_name[tensor.node.layer.name] = shape
        self._output_names = [tensor.node.layer.name for tensor in self.loaded_model.outputs]

    def predict_for_mxa(self, inputs, use_model_shape, pre_model=False, output_order=None):
        """Run inference and handle additional input-output processing for Keras models."""
        if not isinstance(inputs, (dict, list, np.ndarray)):
            raise TypeError(
                "`inputs` must be a mapping from str to np.ndarray, "
                "a list of np.ndarray, or a single np.ndarray."
            )

        # Ensure the number of inputs matches the model's requirements
        assert len(inputs) >= len(self._input_names), (
            f"Expected at least {len(self._input_names)} inputs, but got {len(inputs)}"
        )

        if isinstance(inputs, (list, tuple)):
            if not all(isinstance(x, np.ndarray) for x in inputs):
                raise TypeError('Expected a list of np.ndarray, got: {type(x)}')

        remaining_inputs = {}

        if not use_model_shape:
            # insert batch-dim
            if isinstance(inputs, list):
                inputs = [np.expand_dims(x, 0) for x in inputs]
                inputs = [np.reshape(inputs[i], shape) for i, shape in enumerate(self._input_shapes)]
            elif isinstance(inputs, np.ndarray):
                inputs = [np.expand_dims(inputs, 0)]
                inputs = [np.reshape(inputs[i], shape) for i, shape in enumerate(self._input_shapes)]
            else:
                inputs = {k: np.expand_dims(v, 0) for k, v in inputs.items()}
                for name, fmap in inputs.items():
                    inputs[name] = fmap.reshape(self._input_shapes_by_name[name])
        else:
            if isinstance(inputs, np.ndarray):
                # Single np.ndarray; wrap in a list
                inputs = [inputs]

            # Inputs are named; filter by model input names
            input_names = [tensor.name for tensor in self.loaded_model.inputs]

            if isinstance(inputs, dict):
                # Filter inputs based on model input names
                matched_inputs = {name: inputs[name] for name in input_names if name in inputs}
                remaining_inputs = {key: value for key, value in inputs.items() if key not in input_names}
                inputs = list(matched_inputs.values())

        # Perform inference
        outputs = self.loaded_model(inputs, training=False)

        # Convert outputs to numpy
        if not isinstance(outputs, list):
            outputs = [outputs.numpy()]
        else:
            outputs = [output.numpy() for output in outputs]

        # Postprocess outputs if needed
        if pre_model and use_model_shape is False:
            outputs = [np.squeeze(output, 0) for output in outputs]

        # Append remaining inputs that were not passed to the model
        for key, value in remaining_inputs.items():
            outputs.insert(0, value)

        # Reorder outputs if a specific order is provided
        if output_order is not None:
            assert len(output_order) == len(outputs), (
                f"output_order length ({len(output_order)}) must match outputs length ({len(outputs)})."
            )
            outputs = [outputs[self._output_names.index(name)] for name in output_order]

        return outputs

    @classmethod
    def load(cls, model_path):
        """Load a Keras model from a file."""
        from tf_keras.models import load_model
        import os

        # Check file format and load accordingly
        if os.path.isdir(model_path):
            # Handle SavedModel directory format
            loaded_model = load_model(model_path)
        elif model_path.endswith(".h5") or model_path.endswith(".keras"):
            # Handle Keras HDF5 format
            loaded_model = load_model(model_path)
        else:
            raise ValueError("Unsupported model file format. Use `.h5`, `.keras`, or SavedModel format.")

        # Initialize the model
        model = cls(loaded_model, "keras")
        model._minimal_init(loaded_model, "keras")
        return model

    # Load a Keras model from memory
    @classmethod
    def _from_memory(cls, model_data, weights=None):
        """Load a Keras model from memory (JSON structure and optional weights)."""
        from tf_keras.models import model_from_json

        # Deserialize the model from JSON
        loaded_model = model_from_json(model_data)

        # Load weights if provided
        if weights is not None:
            loaded_model.set_weights(weights)

        # Initialize the model
        model = cls(loaded_model, "keras")
        model._minimal_init(loaded_model, "keras")
        return model

class OnnxModel(Model):
    def _minimal_init(self, loaded_model, model_type):
        """Initialize ONNX model attributes for inference."""
        import onnx
        import onnxruntime
        super()._minimal_init(loaded_model, model_type)

        # Serialize the model to a byte stream for runtime session creation
        fstream = io.BytesIO()
        onnx.save(self.loaded_model, fstream)

        # Configure ONNX runtime session options
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 2  # Parallelism within operations
        sess_options.inter_op_num_threads = 1  # Parallelism across operations

        # Create the inference session
        self.session = onnxruntime.InferenceSession(
            fstream.getvalue(),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Extract input and output details
        self._input_names = [inp.name for inp in self.session.get_inputs()]
        self._input_shapes = [inp.shape for inp in self.session.get_inputs()]
        self._output_names = [out.name for out in self.session.get_outputs()]

        # Replace symbolic batch dimensions with 1
        for shape in self._input_shapes:
            if isinstance(shape[0], str):
                shape[0] = 1

        # Map input names to shapes
        self._input_shapes_by_name = dict(zip(self._input_names, self._input_shapes))

    def _adapt_layout(self, fmap, to_channels_first=True):
        # fmap without batch-dim
        if to_channels_first:
            return np.moveaxis(fmap, -1, 0)
        return np.moveaxis(fmap, 0, -1)

    def _reshape_input(self, fmap, shape):
        fmap = np.expand_dims(self._adapt_layout(fmap), 0)
        if any(isinstance(d, str) for d in shape):  # failed shape inference
            return fmap
        else:
            return fmap.reshape(shape)

    def predict_for_mxa(self, inputs, use_model_shape, pre_model=False, output_order=None):
        """Run inference and handle additional input-output processing for ONNX."""

        # Validate inputs
        if not isinstance(inputs, (dict, list, np.ndarray)):
            raise TypeError(
                "`inputs` must be a mapping from str to np.ndarray, "
                "a list of np.ndarray, or a single np.ndarray"
            )

        assert len(inputs) >= len(self._input_names), (
            f"Expected {len(self._input_names)} inputs, but got {len(inputs)}."
        )

        feed_dict = {}
        remaining_outputs = {}

        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        elif isinstance(inputs, list) and not all(isinstance(fmap, np.ndarray) for fmap in inputs):
            raise TypeError("Expected a list of np.ndarray")

        # Case 1: Input is reshaped to match the model input shape
        if not use_model_shape:
            if isinstance(inputs, dict):
                for name, fmap in inputs.items():
                    feed_dict[name] = self._reshape_input(fmap, self._input_shapes_by_name[name])
            else:
                for i, fmap in enumerate(inputs):
                    feed_dict[self._input_names[i]] = self._reshape_input(fmap, self._input_shapes[i])

        # Case 2: Inputs are directly passed through the model
        else:
            if isinstance(inputs, dict):
                feed_dict = {k: v for k, v in inputs.items() if k in self._input_names}
                remaining_outputs = {k: v for k, v in inputs.items() if k not in self._input_names}
            else:
                for i, arr in enumerate(inputs):
                    feed_dict[self._input_names[i]] = arr

        # Run inference
        outputs = self.session.run(None, feed_dict)

        # Postprocess outputs if needed
        if pre_model and use_model_shape is False:
            processed_outputs = []
            for x in outputs:
                x = np.squeeze(x, axis=0) if x.ndim >= 1 and x.shape[0] == 1 else x  # remove batch dim if it's 1
                if x.ndim == 3:  # (C, H, W) → (H, W, C)
                    x = x.transpose(1, 2, 0)
                processed_outputs.append(x)
            outputs = processed_outputs

        # Insert additional inputs (e.g., passthrough metadata)
        for key, value in remaining_outputs.items():
            outputs.insert(0, value)

        # Reorder outputs if needed
        if output_order is not None:
            assert len(output_order) == len(outputs), (
                f"output_order length ({len(output_order)}) must match outputs length ({len(outputs)})."
            )
            name_to_index = {name: i for i, name in enumerate(self._output_names)}
            outputs = [outputs[name_to_index[name]] for name in output_order]

        return outputs

    @classmethod
    def load(cls, model_path):
        """Load an ONNX model from a file."""
        import onnx

        loaded_model = onnx.load(model_path)
        model = cls(loaded_model, "onnx")
        model._minimal_init(loaded_model, "onnx")
        return model

    @classmethod
    def _from_memory(cls, model_data):
        """Load an ONNX model from memory."""
        import onnx

        loaded_model = onnx.load_model_from_string(model_data)
        model = cls(loaded_model, "onnx")
        model._minimal_init(loaded_model, "onnx")
        return model

class TFLiteModel(Model):
    def _minimal_init(self, loaded_model, model_type):
        """Initialize TFLite model attributes for inference."""
        import tensorflow as tf

        super().__init__(loaded_model, model_type)
        self.interpreter = tf.lite.Interpreter(model_content=self.loaded_model)
        self.interpreter.allocate_tensors()

        # Extract input and output details
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

        # Record input and output names and shapes
        self._input_names = [detail["name"] for detail in self._input_details]
        self._input_shapes = [detail["shape"] for detail in self._input_details]
        self._output_names = [detail["name"] for detail in self._output_details]

    def predict_for_mxa(self, inputs, use_model_shape, pre_model=False, output_order=None):
        """Run inference and handle input-output processing."""

        if not isinstance(inputs, (dict, list, np.ndarray)):
            raise TypeError(
                "`inputs` must be one of - mapping from layer name to np.ndarray, "
                "list of np.ndarray, or a single np.ndarray."
            )

        # Ensure the number of inputs matches the model's requirements
        assert len(inputs) >= len(self._input_details), (
            f"Expected at least {len(self._input_details)} inputs, but got {len(inputs)}"
        )

        remaining_inputs = {}
        # Handle inputs: single ndarray or list
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]

        # Case 1: Input is reshaped to match the model input shape
        if not use_model_shape:
            if isinstance(inputs, dict):
                # Insert batch dimension and typecast
                expanded_inputs = {name: np.expand_dims(arr, 0) for name, arr in inputs.items()}
                inputs.update(expanded_inputs)

                # Set tensors for each input
                for details in self._input_details:
                    dtype = details.get('dtype', np.float32)
                    exp_shape = details['shape']
                    fmap = inputs[details['name']].astype(dtype)

                    if fmap.shape != exp_shape:
                        fmap = fmap.reshape(exp_shape)

                    self.interpreter.set_tensor(details['index'], fmap)

            elif isinstance(inputs, (list, tuple)):
                inputs = [np.expand_dims(arr, 0) for arr in inputs]
                for i, details in enumerate(self._input_details):
                    dtype = details.get('dtype', np.float32)
                    exp_shape = details['shape']
                    fmap = inputs[i].astype(dtype)

                    if fmap.shape != exp_shape:
                        fmap = fmap.reshape(exp_shape)

                    self.interpreter.set_tensor(details['index'], fmap)

        # Case 2: Inputs are directly passed through the model
        else:
            if isinstance(inputs, (list, tuple)):
                if not all(isinstance(x, np.ndarray) for x in inputs):
                    raise TypeError("Expected a list or tuple of np.ndarray.")
                for i, arr in enumerate(inputs):
                    self.interpreter.set_tensor(self._input_details[i]["index"], arr.astype(self._input_details[i]["dtype"]))

            elif isinstance(inputs, np.ndarray):
                self.interpreter.set_tensor(self._input_details[0]["index"], inputs.astype(self._input_details[0]["dtype"]))

            elif isinstance(inputs, dict):
                input_names = [detail["name"] for detail in self._input_details]
                for name, arr in inputs.items():
                    if name in input_names:
                        index = input_names.index(name)
                        self.interpreter.set_tensor(self._input_details[index]["index"], arr.astype(self._input_details[index]["dtype"]))
                    else:
                        remaining_inputs[name] = arr

        # Run inference
        self.interpreter.invoke()

        # Collect outputs
        outputs = []
        for detail in self._output_details:
            output_array = self.interpreter.get_tensor(detail["index"])
            outputs.append(output_array)

        # Postprocess outputs if needed
        if pre_model and use_model_shape is False:
            outputs = [np.squeeze(x, 0) for x in outputs]

        # Add each remaining input to the beginning of outputs
        for key, value in remaining_inputs.items():
            outputs.insert(0, value)

        # Reorder outputs if order is provided
        if output_order:
            assert len(output_order) == len(outputs), (
                f"output_order length ({len(output_order)}) must match "
                f"outputs length ({len(outputs)})."
            )
            outputs = [outputs[self._output_names.index(name)] for name in output_order]

        return outputs

    @classmethod
    def load(cls, model_path):
        """Load TFLite model from a file."""
        with open(model_path, "rb") as f:
            buf_binary = f.read()
        model = cls(buf_binary, "tflite")
        model._minimal_init(buf_binary, "tflite")
        return model

    @classmethod
    def _from_memory(cls, model_data):
        """Load TFLite model from in-memory binary data."""
        model = cls(model_data, "tflite")
        model._minimal_init(model_data, "tflite")
        return model


class TFModel(Model):
    def _minimal_init(self, loaded_model, model_type):
        """Initialize TensorFlow model attributes for inference."""
        import tensorflow as tf
        from tensorflow.python.framework.ops import Graph

        super()._minimal_init(loaded_model, model_type)

        if isinstance(loaded_model, Graph) or hasattr(loaded_model, "inputs"):
            # Case 1: SavedModel or Graph object
            self._input_names = [tensor.name for tensor in loaded_model.inputs]
            self._input_shapes = [list(tensor.shape.as_list()) for tensor in loaded_model.inputs]
            self._output_names = [tensor.name for tensor in loaded_model.outputs]
        elif isinstance(loaded_model, tf.compat.v1.GraphDef):
            # Case 2: Frozen Graph (GraphDef)
            self._input_names = []
            self._input_shapes = []
            self._output_names = []

            # Create a temporary Graph to process the frozen graph
            with tf.compat.v1.Graph().as_default() as graph:
                tf.compat.v1.import_graph_def(loaded_model, name="")
                for op in graph.get_operations():
                    if op.type == "Placeholder":
                        self._input_names.append(op.name)
                        shape = op.outputs[0].shape.as_list()
                        if shape[0] is None or shape[0] == -1:
                            shape[0] = 1  # Default batch size to 1
                        self._input_shapes.append(shape)

                    # Identify output tensors (no consumers)
                    if len(op.outputs) > 0 and len(op.outputs[0].consumers()) == 0:
                        self._output_names.append(op.outputs[0].name)

            # Use the temporary Graph for session-based inference
            self.sess = tf.compat.v1.Session(graph=graph)
        else:
            raise ValueError("Invalid loaded_model type. Expected SavedModel or GraphDef.")

        # Ensure input shapes and names are valid
        if not self._input_names or not self._output_names:
            raise ValueError("Failed to identify valid input/output tensors.")

        self._input_shapes_by_name = dict(zip(self._input_names, self._input_shapes))

        print(f"Input Names: {self._input_names}")
        print(f"Input Shapes: {self._input_shapes}")
        print(f"Output Names: {self._output_names}")

    def predict_for_mxa(self, inputs, use_model_shape, pre_model=False, output_order=None):
        """Run inference and handle input-output processing."""
        if not isinstance(inputs, (dict, list, np.ndarray)):
            raise TypeError("inputs must be a dict, list of np.ndarray, or a single np.ndarray.")

        assert len(inputs) >= len(self._input_names), (
            f"Expected at least {len(self._input_names)} inputs, but got {len(inputs)}"
        )

        # Initialize feed_dict and remaining_inputs safely
        feed_dict = {}
        remaining_inputs = {}

        # Case 1: Input is reshaped to match the model input shape
        if not use_model_shape:
            # Normalize inputs: wrap in lists or dicts with expanded dimensions
            if isinstance(inputs, np.ndarray):
                inputs = [np.expand_dims(inputs, 0)]
            elif isinstance(inputs, list):
                if not all(isinstance(x, np.ndarray) for x in inputs):
                    raise TypeError("Expected list of np.ndarray")
                inputs = [np.expand_dims(x, 0) for x in inputs]
            elif isinstance(inputs, dict):
                inputs = {k: np.expand_dims(v, 0) for k, v in inputs.items()}

            # Create feed_dict with proper reshape
            if isinstance(inputs, dict):
                for name in self._input_names:
                    if name not in inputs:
                        raise KeyError(f"Missing input '{name}' required by model.")
                    # Safely get the shape, or raise a clear error
                    shape = self._input_shapes_by_name.get(name)
                    if shape is None:
                        raise KeyError(f"Shape not found for input '{name}' in _input_shapes_by_name.")
                    feed_dict[name + ':0'] = inputs[name].reshape(shape)

                # Collect any remaining inputs not in the input names
                remaining_inputs = {k: v for k, v in inputs.items() if k not in self._input_names}

            else:
                # For list or tuple types
                for i, name in enumerate(self._input_names):
                    shape = self._input_shapes_by_name.get(name)
                    if shape is None:
                        raise KeyError(f"Shape not found for input '{name}' in _input_shapes_by_name.")
                    feed_dict[name + ':0'] = inputs[i].reshape(shape)

        # Case 2: Inputs are directly passed through the model
        else:
            # Inputs are unnamed; process sequentially
            if isinstance(inputs, (list, tuple)):
                if not all(isinstance(x, np.ndarray) for x in inputs):
                    raise TypeError("Expected a list or tuple of np.ndarray.")
            elif isinstance(inputs, np.ndarray):
                inputs = [inputs]

            # For dictionary inputs
            if isinstance(inputs, dict):
                input_names = self._input_names
                filtered_inputs = {key: inputs[key] for key in input_names if key in inputs}
                remaining_inputs = {key: inputs[key] for key in inputs if key not in input_names}
                feed_dict = {k + ':0': v for k, v in filtered_inputs.items()}

            # For list or ndarray inputs
            elif isinstance(inputs, list):
                for i, name in enumerate(self._input_names):
                    feed_dict[name + ':0'] = inputs[i]
            elif isinstance(inputs, np.ndarray):
                feed_dict[self._input_names[0] + ':0'] = inputs

        # Run inference
        outputs = self.sess.run(self._output_names, feed_dict=feed_dict)

        # Postprocess if needed
        if pre_model and use_model_shape is False:
            outputs = [
                x.squeeze(0).transpose(1, 2, 0) if x.ndim == 4 and x.shape[0] == 1 else x
                for x in outputs
            ]

        # Attach remaining metadata inputs
        for key, value in remaining_inputs.items():
            outputs.insert(0, value)

        # Reorder outputs if specified
        if output_order:
            name_to_index = {k: i for i, k in enumerate(self._output_names)}
            outputs = [outputs[name_to_index[name + ':0']] for name in output_order]

        return outputs

    @classmethod
    def load(cls, model_path):
        """Load a TensorFlow model from a file."""
        import tensorflow as tf

        if os.path.isdir(model_path):
            # Load SavedModel
            loaded_model = tf.saved_model.load(model_path)
        elif os.path.isfile(model_path) and model_path.endswith(".pb"):
            # Load Frozen Graph
            graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def.ParseFromString(f.read())
            loaded_model = graph_def
        else:
            raise ValueError(f"Invalid model path: {model_path}")

        model = cls(loaded_model, {})
        model._minimal_init(loaded_model, "tf")
        return model

    @classmethod
    def _from_memory(cls, model_data):
        """Load a TensorFlow model from memory."""
        import tensorflow as tf

        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(model_data)

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        model = cls(None, "tf")
        model.graph = graph
        model.sess = tf.compat.v1.Session(graph=graph)
        model._minimal_init(None, "tf")
        return model


class ModelLoader:
    def __init__(self):
        # Mapping of file extensions to model types
        self._extension_to_model_type = {
            ".h5": "keras",
            ".onnx": "onnx",
            ".tflite": "tflite",
            ".pb": "tf",
        }

        # Factory mapping model types to their respective loader classes
        self._model_loader_factory = {
            "keras": KerasModel,
            "onnx": OnnxModel,
            "tflite": TFLiteModel,
            "tf": TFModel,
        }

    def load(self, model_path):
        """Load a model from a file by inferring the model type from the extension."""
        if not isinstance(model_path, str):
            raise TypeError("`model_path` must be a string representing the file path.")

        # Get the file extension and determine the model type
        file_extension = Path(model_path).suffix.lower()
        model_type = self._extension_to_model_type.get(file_extension)

        if not model_type or model_type not in self._model_loader_factory:
            raise ValueError(f"Unsupported model file extension: {file_extension}")

        # Use the factory to load the model
        return self._model_loader_factory[model_type].load(model_path)

    def from_memory(self, model_data):
        """Load a model from memory."""
        if isinstance(model_data, dict) and "model_type" in model_data:
            model_type = model_data["model_type"]
            model_content = model_data["content"]

            if model_type not in self._model_loader_factory:
                raise ValueError(f"Unsupported model type in memory: {model_type}")

            # Use the factory to load the model from memory
            return self._model_loader_factory[model_type]._from_memory(model_content)
        else:
            raise TypeError(
                "Expected `model_data` to be a dictionary with 'model_type' and 'content' keys."
            )
