#ifndef TORCHVECTOR_HH
#define TORCHVECTOR_HH

#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace torch;
/*
This class is a shallow wrapper around torch::Tensor. It is used to allow
the torch vector to be used in the same way as the ThreeVector class.
*/

namespace py = pybind11;


class TorchVector: public Tensor
{
public:
    // Default constructor
    TorchVector(): Tensor(){};

    TorchVector(Tensor tensor) : Tensor(std::move(tensor)) {}

    TorchVector(Tensor tensor, bool requires_grad)
    {
        *this = Tensor(std::move(tensor));
        this->set_requires_grad(requires_grad);
    }

    // Constructor for consisency with torch. just takes an initializer list and passes it to the torch::tensor init
    TorchVector(const py::list& initializerList)
    {
        assert (initializerList.size() == 3);
        auto vec = initializerList.cast<std::vector<float>>();
        *this = torch::tensor(vec);
    }

    // Constructor for consisency with torch. just takes an initializer list and passes it to the torch::tensor init
    TorchVector(const py::list& initializerList, bool requires_grad)
    {
        assert (initializerList.size() == 3);
        auto vec = initializerList.cast<std::vector<float>>();
        *this = torch::tensor(vec);
        this->set_requires_grad(requires_grad);
    }

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

    // Bad init constructor
	TorchVector(float* ptr)
	{
        *this = tensor({ptr[0], ptr[1], ptr[2]});
	}

    float getItem(size_t index) const
    {
        return data_ptr<float>()[index];
    }

    TorchVector Magnitude() const
    {
        return torch::norm(*this, 2);
    }

    TorchVector Magnitude2() const
    {
        return torch::square(torch::norm(*this, 2));
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