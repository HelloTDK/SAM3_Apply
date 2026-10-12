"""ACL model buffers and synchronous execution; no PyTorch dependency."""

import acl
import numpy as np


# Some ACL Python releases omit the aclDataType enum constants.
ACL_DTYPES = {
    0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32, 4: np.uint8,
    6: np.int16, 7: np.uint16, 8: np.uint32, 9: np.int64, 10: np.uint64,
    11: np.float64, 12: np.bool_,
}


def check_ret(operation, ret):
    if ret != 0:
        raise RuntimeError(f"{operation} failed, ret={ret}")


class ACLRuntime:
    def __init__(self, device_id):
        self.device_id = device_id
        self.context = None
        check_ret("acl.init", acl.init())
        try:
            check_ret("acl.rt.set_device", acl.rt.set_device(device_id))
            self.context, ret = acl.rt.create_context(device_id)
            check_ret("acl.rt.create_context", ret)
        except Exception:
            acl.rt.reset_device(device_id)
            acl.finalize()
            raise

    def bind_context(self):
        check_ret("acl.rt.set_context", acl.rt.set_context(self.context))

    def close(self):
        if self.context is None:
            return
        self.bind_context()
        check_ret("acl.rt.destroy_context", acl.rt.destroy_context(self.context))
        self.context = None
        check_ret("acl.rt.reset_device", acl.rt.reset_device(self.device_id))
        check_ret("acl.finalize", acl.finalize())


class ACLModel:
    def __init__(self, path):
        self.path = str(path)
        self.model_id = None
        self.desc = None
        self.output_dataset = None
        self.output_buffers = []
        self.output_data_buffers = []
        self.model_id, ret = acl.mdl.load_from_file(self.path)
        check_ret(f"acl.mdl.load_from_file: {path}", ret)
        try:
            self.desc = acl.mdl.create_desc()
            check_ret("acl.mdl.get_desc", acl.mdl.get_desc(self.desc, self.model_id))
            self.in_shapes, self.in_dtypes, self.in_sizes = self._specs("input")
            self.out_shapes, self.out_dtypes, self.out_sizes = self._specs("output")
            self.output_dataset = acl.mdl.create_dataset()
            for size in self.out_sizes:
                buffer, ret = acl.rt.malloc(size, 0)
                check_ret("acl.rt.malloc output", ret)
                self.output_buffers.append(buffer)
                data_buffer = acl.create_data_buffer(buffer, size)
                self.output_data_buffers.append(data_buffer)
                _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)
                check_ret("acl.mdl.add_dataset_buffer output", ret)
        except Exception:
            self.close()
            raise

    def _specs(self, kind):
        count = getattr(acl.mdl, f"get_num_{kind}s")(self.desc)
        shapes, dtypes, sizes = [], [], []
        for index in range(count):
            dims, ret = getattr(acl.mdl, f"get_{kind}_dims")(self.desc, index)
            check_ret(f"acl.mdl.get_{kind}_dims", ret)
            shape = tuple(dims["dims"])
            dtype = np.dtype(ACL_DTYPES[getattr(acl.mdl, f"get_{kind}_data_type")(self.desc, index)])
            size = getattr(acl.mdl, f"get_{kind}_size_by_index")(self.desc, index)
            if any(dim <= 0 for dim in shape) or np.prod(shape) * dtype.itemsize != size:
                raise ValueError(f"{self.path}: unsupported {kind} shape/dtype/size at index {index}")
            shapes.append(shape)
            dtypes.append(dtype)
            sizes.append(size)
        return shapes, dtypes, sizes

    def run(self, inputs):
        if len(inputs) != len(self.in_shapes):
            raise ValueError(f"{self.path}: expected {len(self.in_shapes)} inputs, got {len(inputs)}")
        arrays = []
        for index, (value, shape, dtype) in enumerate(zip(inputs, self.in_shapes, self.in_dtypes)):
            array = np.ascontiguousarray(value, dtype=dtype)
            if array.shape != shape:
                raise ValueError(f"{self.path}: input {index} expects {shape}, got {array.shape}")
            arrays.append(array)
        dataset = acl.mdl.create_dataset()
        buffers, data_buffers = [], []
        try:
            for array, size in zip(arrays, self.in_sizes):
                buffer, ret = acl.rt.malloc(size, 0)
                check_ret("acl.rt.malloc input", ret)
                buffers.append(buffer)
                raw = array.tobytes()
                pointer = acl.util.bytes_to_ptr(raw)
                check_ret("acl.rt.memcpy H2D", acl.rt.memcpy(buffer, size, pointer, size, 1))
                data_buffer = acl.create_data_buffer(buffer, size)
                data_buffers.append(data_buffer)
                _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
                check_ret("acl.mdl.add_dataset_buffer input", ret)
            check_ret(f"acl.mdl.execute: {self.path}", acl.mdl.execute(self.model_id, dataset, self.output_dataset))
            outputs = []
            for buffer, size, shape, dtype in zip(self.output_buffers, self.out_sizes, self.out_shapes, self.out_dtypes):
                host, ret = acl.rt.malloc_host(size)
                check_ret("acl.rt.malloc_host", ret)
                try:
                    check_ret("acl.rt.memcpy D2H", acl.rt.memcpy(host, size, buffer, size, 2))
                    raw = acl.util.ptr_to_bytes(host, size)
                    outputs.append(np.frombuffer(raw, dtype=dtype).reshape(shape).copy())
                finally:
                    acl.rt.free_host(host)
            return outputs
        finally:
            for data_buffer in data_buffers:
                acl.destroy_data_buffer(data_buffer)
            acl.mdl.destroy_dataset(dataset)
            for buffer in buffers:
                acl.rt.free(buffer)

    def close(self):
        for data_buffer in self.output_data_buffers:
            acl.destroy_data_buffer(data_buffer)
        self.output_data_buffers.clear()
        if self.output_dataset is not None:
            acl.mdl.destroy_dataset(self.output_dataset)
            self.output_dataset = None
        for buffer in self.output_buffers:
            acl.rt.free(buffer)
        self.output_buffers.clear()
        if self.desc is not None:
            acl.mdl.destroy_desc(self.desc)
            self.desc = None
        if self.model_id is not None:
            check_ret("acl.mdl.unload", acl.mdl.unload(self.model_id))
            self.model_id = None
