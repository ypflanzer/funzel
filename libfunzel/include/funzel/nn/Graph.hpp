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

#include <optional>
#include <iostream>
#include <unordered_map>

namespace funzel
{
namespace nn
{

class IGraphNode
{
public:
	virtual void dump(std::ostream& out) = 0;
	virtual void execute() = 0;

	class GraphNodeResult
	{
	public:
		GraphNodeResult(IGraphNode* node, Tensor* tensor):
			node(node), tensor(tensor) {}

		const Tensor& get()
		{
			//std::cout << tensor << ": " << *tensor << std::endl;
			if(tensor->size() == 0)
				node->execute();
			return *tensor;
		}

	private:
		IGraphNode* node;
		Tensor* tensor = nullptr;
	};

	virtual GraphNodeResult result(size_t idx = 0) = 0;

	typedef std::unordered_map<std::string, GraphNodeResult> ResultSet;
	virtual ResultSet getResults() { return {{"value", result()}};}

	operator GraphNodeResult()
	{
		return result(0);
	}
};

typedef std::shared_ptr<IGraphNode> IGraphNodeRef;

class ConstantNode : public IGraphNode
{
public:
	void dump(std::ostream& out) override {}
	void execute() override {}

	Tensor& value() { return m_value; }
	GraphNodeResult result(size_t idx = 0) override
	{
		return GraphNodeResult{this, &m_value};
	}

private:
	Tensor m_value;
};

template<typename Fn>
class BinaryOpNode : public IGraphNode
{
public:
	BinaryOpNode(const GraphNodeResult& a, const GraphNodeResult& b):
		a(a), b(b) {}

	void dump(std::ostream& out) override {}
	void execute() override
	{
		m_result = Fn().operator()(a.get(), b.get());
	}

	GraphNodeResult result(size_t idx = 0) override
	{
		return GraphNodeResult{this, &m_result};
	}

private:
	GraphNodeResult a, b;
	Tensor m_result;
};

typedef BinaryOpNode<decltype([](const Tensor& a, const Tensor& b) { return a - b; })> SubNode;
typedef BinaryOpNode<decltype([](const Tensor& a, const Tensor& b) { return a + b; })> AddNode;
typedef BinaryOpNode<decltype([](const Tensor& a, const Tensor& b) { return a * b; })> MulNode;
typedef BinaryOpNode<decltype([](const Tensor& a, const Tensor& b) { return a / b; })> DivNode;

template<typename Fn>
class UnaryOpNode : public IGraphNode
{
public:
	UnaryOpNode(const GraphNodeResult& x):
		x(x) {}

	void dump(std::ostream& out) override {}
	void execute() override
	{
		m_result = Fn().operator()(x.get());
	}

	GraphNodeResult result(size_t idx = 0) override
	{
		return GraphNodeResult{this, &m_result};
	}

private:
	GraphNodeResult x;
	Tensor m_result;
};

typedef UnaryOpNode<decltype([](const Tensor& x) { return x.sin(); })> SinNode;
typedef UnaryOpNode<decltype([](const Tensor& x) { return x.cos(); })> CosNode;
typedef UnaryOpNode<decltype([](const Tensor& x) { return x.tan(); })> TanNode;

}
}
