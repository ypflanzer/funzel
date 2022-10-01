#include <funzel/cv/Image.hpp>
#include <funzel/Plot.hpp>

using namespace funzel;

class Plot1D : public Subplot
{
public:
	void serialize(std::ostream& out, unsigned int index) const override
	{
		const auto& t = data();
		AssertExcept(t.shape.size() == 1 || t.shape.size() == 2, "Expected linear data.");

		out << "$data" << index << " <<EOD\n";
		if(t.shape.size() == 1)
		{
			for(size_t i = 0; i < t.shape[0]; i++)
			{
				out << i << " " << t[i].item<double>() << "\n";
			}
		}
		else
		{
			for(size_t i = 0; i < t.shape[0]; i++)
			{
				out << t[i][0].item<double>() << " " << t[i][1].item<double>() << "\n";
			}
		}
		out << "EOD\n";
		out << (index > 0 ? "re" : "") << "plot $data" << index << " w " << shape() << " lc '" << color() << "' title '" << title() << "'\n";
	}
};

namespace funzel
{
std::ostream& operator<<(std::ostream& out, const Subplot& s)
{
	s.serialize(out);
	return out;
}
}

static std::string findGnuplot()
{
	char* envvar = getenv("FUNZEL_GNUPLOT");
	if(envvar)
		return envvar;

	return "gnuplot"; // TODO Real search!!!
}

#include <iostream>
#include <filesystem>

#ifdef WIN32
#define popen _popen
#define pclose _pclose
#endif

void Plot::sendToGnuplot(const std::string& code)
{
	const char* args = " -p";
	const auto gnuplot = findGnuplot();
	auto* pipe = popen((gnuplot + args).c_str(), "w");
	AssertExcept(pipe, "Could not launch gnuplot!");
	
	fwrite(code.c_str(), code.size(), 1, pipe);
	pclose(pipe);
}

void Plot::serialize(std::ostream& out) const
{
	out << "set title '" << m_title << "'\n";

	unsigned int idx = 0;
	for(auto& plot : m_subplots)
	{
		plot->serialize(out, idx++);
		out << '\n';
	}
}

void Plot::show(bool wait)
{
	std::stringstream ss;
	// ss << "set terminal wxt\n";
	serialize(ss);
	sendToGnuplot(ss.str());
}

void Plot::save(const std::string& file)
{
	std::stringstream ss;

	const size_t extIdx = file.find_last_of('.');
	AssertExcept(extIdx != std::string::npos, "Cannot infer image type from path: " << file);

	ss << "set terminal " << file.substr(extIdx + 1) << '\n'
	   << "set output '" << file << "'\n";
	
	serialize(ss);
	sendToGnuplot(ss.str());
}

std::shared_ptr<Subplot> Plot::plot(const Tensor& t, const std::string& title)
{
	auto subplot = std::make_shared<Plot1D>();
	subplot->title(title).data(t);
	m_subplots.push_back(subplot);

	return subplot;
}

class PlotImage : public Subplot
{
public:
	void serialize(std::ostream& out, unsigned int index) const override
	{
		const auto& t = data();
		AssertExcept(t.shape.size() == 2 || t.shape.size() == 3, "Expected image data with WHC.");

		// TODO Proper temporary directory!
		std::string imageName = "data" + std::to_string(index) + ".png";
		funzel::image::save(data(), imageName);
		
		out << "set size ratio -1\n"
			<< "set yrange [0 : " << t.shape[0] << "]\n"
			<< "set xrange [0 : " << t.shape[1] << "]\n"
			<< (index > 0 ? "re" : "") << "plot '" << imageName << "' binary filetype=png w rgbimage title '" << title() << "'\n";
	}
};

std::shared_ptr<Subplot> Plot::image(const Tensor& t, const std::string& title)
{
	auto subplot = std::make_shared<PlotImage>();
	subplot->title(title).data(t);
	m_subplots.push_back(subplot);

	return subplot;
}

