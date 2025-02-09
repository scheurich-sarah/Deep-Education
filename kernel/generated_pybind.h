//#include <pybind11/pybind11.h>
//#include "dlpack.h"
//#include "kernel.h"
//#include "csr.h"

//namespace py = pybind11;


// expects python capsule for output
// convert to simple 1D or 2D array
// once inside, calls invoke_gspmm
inline void export_kernel(py::module &m) { 
    m.def("gspmm",
        [](graph_t& graph, py::capsule& input, py::capsule& output, bool reverse, bool norm) {
            array2d_t<float> input_array = capsule_to_array2d(input);
            array2d_t<float> output_array = capsule_to_array2d(output);
            return invoke_gspmm(graph, input_array, output_array, reverse, norm);
        }
    );

    // for multithreaded version
    m.def("gspmm_mt",
        [](graph_t& graph, py::capsule& input, py::capsule& output, bool reverse, bool norm) {
            array2d_t<float> input_array = capsule_to_array2d(input);
            array2d_t<float> output_array = capsule_to_array2d(output);
            return invoke_gspmm_mt(graph, input_array, output_array, reverse, norm);
        }
    );
}
