from setuptools import setup, dist
from setuptools.command.install import install
import os
import subprocess

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "../README.md")) as f:
	long_description = f.read()

class BinaryDistribution(dist.Distribution):
	def has_ext_modules(foo):
		return True

class PostInstallCommand(install):
	def run(self):
		install.run(self)

		if not os.path.isdir(self.install_scripts):
			os.makedirs(self.install_scripts)

		cwd = this_directory
		builddir = os.path.join(cwd, "pipbuild")
		pkgdir = os.path.join(self.install_platlib, "funzel") # os.path.join(cwd, "funzel")
		os.makedirs(builddir, exist_ok=True)

		configCmd = ["cmake", "-S", "..", "-B", builddir, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX='{pkgdir}'", "-DUSE_RELATIVE_RPATH=TRUE"]
		buildCmd = ["cmake", "--build", builddir, "--target", "install", "--config", "Release"]

		subprocess.check_call(configCmd)
		subprocess.check_call(buildCmd)

setup(
	name="funzel",
	version="0.0.1",
	description="The Funzel tensor library.",
	url="https://github.com/sponk/funzel",
	author="Yannick Pflanzer",
	author_email="",
	license="LGPLv3",
	packages=["funzel"],
	install_requires=[],
	
	long_description=long_description,
	long_description_content_type="text/markdown",
	include_package_data=True,
	distclass=BinaryDistribution,
	cmdclass={'install': PostInstallCommand},

	classifiers=[
		"Development Status :: 1 - Planning",
		"Intended Audience :: Science/Research",
		"License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
		"Operating System :: POSIX :: Linux",
		"Operating System :: Microsoft :: Windows",
		"Operating System :: MacOS :: MacOS X",
		"Programming Language :: Python :: 2",
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.10",
		"Programming Language :: Python :: 3.12"
	],
)

