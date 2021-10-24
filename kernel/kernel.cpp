#include <cassert>
#include <iostream>
#include <limits>

#include "kernel.h"

using std::cout;
using std::endl;

int THD_COUNT = 1;

using std::string;


void _gspmm(csr_t* snaph, array2d_t<float> & input, array2d_t<float> & output, 
                     op_t op, bool reverse, bool norm /*= true*/)
{
    //cout << "spmm " << op << "reverse = " << reverse << endl;

    //If in backward, normalize it first, else normalize it after computation
    
    //The core logic goes here.    
    // needs to be independent of any tensor data struc
    // GAS in this case: gather all neighbors' messgages/tensors, then sum
    // each vertex has a feature vec, when you sum it, you get another vec
}

// The signature for this function is in kernel.h
// very simple func,just pass a couple things
// one of which needs to be a pointer
// reverse: indicates whether or not to do forward or backward compute
void invoke_gspmm(graph_t& graph, array2d_t<float> & input_array, array2d_t<float> & output_array,
                 bool reverse, bool norm /*= true*/)
{
    if (reverse) {
	 // backward computation uses csr
	 // normalzing involves dividing each input param/tensor
	 // by it's degree, which you need to calc
         return _gspmm(&graph.csr, input_array, output_array, eSUM, reverse, norm);
    } else {
	 // forward computation uses csc, the transpose of csr
	 // do GAS then normalize
         return _gspmm(&graph.csc, input_array, output_array, eSUM, reverse, norm);
    }

}
