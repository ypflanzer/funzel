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

#include "Tensor.hpp"

namespace funzel
{

class Subplot
{
public:
	Subplot& title(const std::string& str)
	{
		m_title = str;
		return *this;
	}

	Subplot& data(const Tensor& t)
	{
		m_data = t;
		return *this;
	}

	Subplot& shape(const std::string& str)
	{
		m_shape = str;
		return *this;
	}

	Subplot& color(const std::string& str)
	{
		m_color = str;
		return *this;
	}

	const std::string& title() const { return m_title; }
	const Tensor& data() const { return m_data; }
	const std::string& shape() const { return m_shape; }
	const std::string& color() const { return m_color; }

	virtual void serialize(std::ostream& out, unsigned int index = 0) const = 0;

private:
	std::string m_title, m_shape = "points", m_color = "blue";
	Tensor m_data;
};

class Plot
{
public:
	void show(bool wait = true);
	std::shared_ptr<Subplot> plot(const Tensor& t, const std::string& title = EmptyStr);
	std::shared_ptr<Subplot> image(const Tensor& t, const std::string& title = EmptyStr);

	Plot& title(const std::string& str)
	{
		m_title = str;
		return *this;
	}

private:
	std::vector<std::shared_ptr<Subplot>> m_subplots;
	std::string m_title;
};

}
