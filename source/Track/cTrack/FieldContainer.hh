#ifndef FieldContainer_HH
#define FieldContainer_HH

#include "ThreeVector.hh"

class FieldContainer
{
public:
	FieldContainer();

	~FieldContainer();

	ThreeVector getField(const ThreeVector& position){
		return ThreeVector(0, 175.63117453347965, 0);
	};
	
};

#endif