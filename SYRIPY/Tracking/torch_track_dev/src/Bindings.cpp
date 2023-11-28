#include <pybind11/pybind11.h>
#include "PtTrack.hh"
#include "Field.hh"

#ifdef USE_TORCH
    #include "TorchVector.hh"
    #include <torch/torch.h>
    #include <torch/csrc/autograd/variable.h>
    #include <torch/csrc/autograd/function.h>
    typedef float scalarType;
    typedef TorchVector vectorType;
#else
    #include "ThreeVector.hh"
    typedef double scalarType;
    typedef ThreeVector vectorType;
#endif

namespace py = pybind11;


PYBIND11_MODULE(cTrack, module)
{
    py::class_<PtTrack>(module, "PtTrack")
        .def(py::init<>())
        .def("simulateTrack", &PtTrack::simulateTrack, "Main function to"
            "simulate the track of a single electron.")
        .def("test", &PtTrack::test, "test function")
        .def("simulateBeam", &PtTrack::simulateBeam, "Main function to"
            "simulate a beam of particles.")
        .def("setTime", &PtTrack::setTime, "Sets the time parameters for "
        	"the simulation")
        .def("setCentralInit", &PtTrack::setCentralInit, "Sets the initial parameters fo r"
        	"the simulation")
        .def("setBeamInit", &PtTrack::setBeamInit, "Sets the beam parameters."
            "for the simulation")
        .def("setField", &PtTrack::setField, "Sets the field container for the"
            "solver.");

    py::class_<FieldContainer>(module, "FieldContainer")
        .def(py::init<>())
        .def("addElement", &FieldContainer::addElement, "Adds a field element"
            "to the container.");

    py::class_<vectorType>(module, "ThreeVector")
        .def(py::init<std::initializer_list<scalarType>>())
        .def(py::init<std::initializer_list<scalarType>, bool>());
}