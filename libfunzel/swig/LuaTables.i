%{
// reads a table into a vector of numbers
// lua numbers will be cast into the type required (rounding may occur)
// return 0 if non numbers found in the table
// returns new'ed ptr if ok
template<class T>
std::vector<T>* SWIG_read_number_vector(lua_State* L, int index)
{
	int i=0;
	size_t len = lua_rawlen(L, index);
	std::vector<T>* vec = new std::vector<T>();
	vec->reserve(len);

	while(1)
	{
		lua_rawgeti(L, index, i+1);
		if (lua_isnil(L,-1))
		{
			lua_pop(L, 1);
			break;	// finished
		}

		if (!lua_isnumber(L,-1))
		{
			lua_pop(L,1);
			delete vec;

			std::cout << "ERROR: Table contains non-numerical values!" << std::endl;
			return 0;	// error
		}

		vec->push_back((T) lua_tonumber(L,-1));
		lua_pop(L,1);
		++i;
	}

	return vec; // ok
}
// writes a vector of numbers out as a lua table
template<class T>
int SWIG_write_number_vector(lua_State* L,std::vector<T> *vec)
{
	lua_newtable(L);
	for(int i=0;i<vec->size();++i)
	{
		lua_pushnumber(L,(double)((*vec)[i]));
		lua_rawseti(L,-2,i+1);// -1 is the number, -2 is the table
	}

	return 0;
}
%}
// then the typemaps
%define SWIG_TYPEMAP_NUM_VECTOR(T)
// in
%typemap(in) std::vector<T> *INPUT
%{	$1 = SWIG_read_number_vector<T>(L,$input);
	if (!$1) SWIG_fail;%}
%typemap(freearg) std::vector<T> *INPUT
%{	delete $1;%}
// out
%typemap(argout) std::vector<T> *OUTPUT
%{	SWIG_write_number_vector(L,$1); SWIG_arg++; %}
%enddef

%define SWIG_TYPEMAP_NUM_CUSTOM(C, T)
// in
%typemap(in) C
%{	$1 = SWIG_read_number_vector<T>(L,$input);
	if (!$1) SWIG_fail;%}
%typemap(freearg) C
%{	delete $1;%}
// out
%typemap(argout) C
%{	SWIG_write_number_vector(L,$1); SWIG_arg++; %}

%typemap(in) C INPUT
%{	$1 = SWIG_read_number_vector<T>(L,$input);
	if (!$1) SWIG_fail;%}
%typemap(freearg) C INPUT
%{	delete $1;%}
// out
%typemap(argout) C OUTPUT
%{	SWIG_write_number_vector(L,$1); SWIG_arg++; %}
%enddef

