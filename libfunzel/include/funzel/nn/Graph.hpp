/* 
 * This file is part of Funzel.
 * Copyright (c) 2022 Yannick Pflanzer.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#pragma once

#include "Module.hpp"
#include "Sequential.hpp"

namespace funzel
{
namespace nn
{

class IGraphNode : public Module
{
public:
	virtual void dump(std::ostream& out) = 0;
};

template<typename T>
class GraphNode : public IGraphNode
{
public:
	GraphNode() = default;
	GraphNode(T&& t) : operation(std::move(t)) {}
	GraphNode(const T& t): operation(t) {}

	~GraphNode() = default;

	void dump(std::ostream& out) override
	{
		out << "rectangle \"" << operation.name() << '@' << this <<  "\"\n";

		for(const auto& v : children)
		{
			v->dump(out);
			out << '"' << operation.name() << '@' << this
				<< "\" -down-> \"" << v->name() << '@' << v.get() << "\"\n";
		} 
	}

	Tensor forward(const Tensor& in) override
	{
		return {};
	}

	Tensor backward(const Tensor& in) override
	{
		return {};
	}

	template<typename T, typename... Args>
	std::shared_ptr<GraphNode<T>> add(Args&&... args)
	{
		std::shared_ptr<GraphNode<T>> g(new nn::GraphNode<T>({args...}));
		children.push_back(g);
		return g;
	}

	void add(std::shared_ptr<GraphNode<T>> n)
	{
		children.push_back(n);
	}

	const char* name() const { return operation.name(); }

	T operation;
	std::vector<std::shared_ptr<IGraphNode>> children;
};

class FUNZEL_API Graph : public Module
{
public:
	Graph() = default;

	Tensor forward(const Tensor& input) final override;
	Tensor backward(const Tensor& input) final override;
	void to(const std::string& device = EmptyStr) final override;
	void defaultInitialize() final override;

private:
};

}
}
