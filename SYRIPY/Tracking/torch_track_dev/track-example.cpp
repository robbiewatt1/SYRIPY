#include <torch/torch.h>
#include <iostream>
#include "Field.hh"
#include "PtTrack.hh"
#include <chrono>
#include "ThreeVector.hh"
//#include "TorchVector.hh"

typedef ThreeVector TorchVector;

int main(int argc, char** argv)
{

    Dipole field = Dipole(TorchVector({0., 0., 0.}),
         TorchVector({0., 0., 5.}), 1.e6, 1.);

    FieldContainer field_con = FieldContainer(std::vector<FieldBlock*>{&field});

    TorchVector position = TorchVector({0., 0., 0.});
    TorchVector momentum = TorchVector({0., 1., 0.});

    PtTrack track = PtTrack();
    track.setField(field_con);
    track.setTime(0., 5., 1000);
    track.setCentralInit(position, momentum);
    track.simulateTrack();
}
