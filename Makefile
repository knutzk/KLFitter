# Copyright (c) 2009--2017, the KLFitter developer team
#
# This file is part of KLFitter.
#
# KLFitter is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or (at
# your option) any later version.
#
# KLFitter is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with KLFitter. If not, see <http://www.gnu.org/licenses/>.
#
INCDIR = include
SRCDIR = src
OBJDIR = obj
LIBDIR = lib
DESTDIR = dest-tmp

CXX = g++
MKDIR = mkdir -p
RM = rm -f
CP = cp -r

ROOTCFLAGS = $(shell root-config --cflags)
ROOTLIBS   = $(shell root-config --libs) -lMinuit

BATCFLAGS = -I$(BATINSTALLDIR)/include
BATLIBS   = -L$(BATINSTALLDIR)/lib -lBAT

SRC = $(wildcard $(SRCDIR)/*.cxx)
OBJ = $(SRC:$(SRCDIR)/%.cxx=$(OBJDIR)/%.o)
MAIN = $(wildcard *.c)
LIBSO = $(LIBDIR)/libKLFitter.so

SOFLAGS = -shared
CXXFLAGS = $(ROOTCFLAGS) $(BATCFLAGS) -I$(INCDIR) -Wall -Wno-deprecated -O2 -ggdb -g -fPIC
LIBS     = $(ROOTLIBS) $(BATLIBS)

# rule for shared library
$(LIBSO): $(OBJ)
	@if [ ! -e $(LIBDIR) ]; then $(MKDIR) $(LIBDIR); fi
	$(CXX) $(CXXFLAGS) $(LIBS) $(SOFLAGS) $+ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cxx
	@if [ ! -e $(OBJDIR) ]; then $(MKDIR) $(OBJDIR); fi
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: all

all: $(LIBSO)

.PHONY: clean

clean:
	$(RM) $(OBJ) $(LIBSO)
	$(RM) -r $(DESTDIR)

.PHONY: install

install: all
	$(MKDIR) $(DESTDIR)/include
	$(MKDIR) $(DESTDIR)/lib
	$(CP) $(LIBDIR) $(DESTDIR)/
	$(CP) $(INCDIR) $(DESTDIR)/
