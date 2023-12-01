#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Track.hh"
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
    py::class_<Track>(module, "Track")
        .def(py::init<>())
        .def("simulateTrack", &Track::simulateTrack, "Main function to"
            "simulate the track of a single electron.")
        .def("backwardTrack", &Track::backwardTrack, "Function for performing"
            "backpropagation of the track.")
        .def("simulateBeam", &Track::simulateBeam, "Main function to"
            "simulate a beam of particles.")
        .def("setTime", &Track::setTime, "Sets the time parameters for "
        	"the simulation")
        .def("setCentralInit", &Track::setCentralInit, "Sets the initial parameters fo r"
        	"the simulation")
        .def("setBeamInit", &Track::setBeamInit, "Sets the beam parameters."
            "for the simulation")
        .def("setField", &Track::setField, "Sets the field container for the"
            "solver.");

    py::class_<FieldContainer>(module, "FieldContainer")
        .def(py::init<>())
        .def("addElement", &FieldContainer::addElement, "Adds a field element"
            "to the container.");

    py::class_<vectorType>(module, "ThreeVector")
        .def(py::init<py::list>())
        .def(py::init<py::list, bool>());
}