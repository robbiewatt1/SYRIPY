#ifndef TORCHVECTOR_HH
#define TORCHVECTOR_HH

#include <torch/torch.h>

using namespace torch;
/*
This class is a shallow wrapper around torch::Tensor. It is used to allow
the torch vector to be used in the same way as the ThreeVector class.
*/

class TorchVector: public Tensor
{
public:
    // Default constructor
    TorchVector(): Tensor(){};

    TorchVector(Tensor tensor) : Tensor(std::move(tensor)) {}


    // Constructor for consisency with torch. just takes an initializer list and passes it to the torch::tensor init
    TorchVector(std::initializer_list<float> initializerList)
    {
        assert (initializerList.size() == 3);
        *this = tensor(initializerList);
    }

    // same as above but with torch::requires_grad() type added
    TorchVector(std::initializer_list<float> initializerList, bool requires_grad)
    {
        assert (initializerList.size() == 3);
        *this = tensor(initializerList);
        this->set_requires_grad(requires_grad);
    }

    float getItem(size_t index) const
    {
        return data_ptr<float>()[index];
    }

    TorchVector Magnitude() const
    {
        return torch::norm(*this);
    }

    TorchVector Magnitude2() const
    {
        return torch::norm(*this, 2);
    }

    TorchVector Unit() const
    {
        return *this / torch::norm(*this);
    }

    TorchVector Cross(const TorchVector& vector) const
    {
        return torch::cross(*this, vector, 0);
    }
};

#endif // THREEVECTOR_HH