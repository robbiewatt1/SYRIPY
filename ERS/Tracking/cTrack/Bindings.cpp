#include <pybind11/pybind11.h>
#include "Track.hh"
#include "Field.hh"
#include "ThreeVector.hh"


namespace py = pybind11;


PYBIND11_MODULE(cTrack, module)
{
    py::class_<Track>(module, "Track")
        .def(py::init<>())
        .def("simulateTrack", &Track::simulateTrack, "Main function to"
            "simulate the track of a single electron.")
        .def("simulateBeam", &Track::simulateBeam, "Main function to"
            "simulate a beam of particles.")
        .def("setTime", &Track::setTime, "Sets the time parameters for "
        	"the simulation")
        .def("setCentralInit", &Track::setCentralInit, "Sets the initial parameters fo r"
        	"the simulation")
        .def("setBeamParams", &Track::setBeamParams, "Sets the beam parameters."
            "for the simulation")
        .def("setField", &Track::setField, "Sets the field container for the"
            "solver.");

    py::class_<FieldContainer>(module, "FieldContainer")
        .def(py::init<>())
        .def("addElement", &FieldContainer::addElement, "Adds a field element"
            "to the container.");

    py::class_<ThreeVector>(module, "ThreeVector")
        .def(py::init<double, double, double>());
}