#  MinGW
ifeq "$(OS)" "Windows_NT"
CFLG=-O3 -Wall -I.
LIBS=-lfreeglut -lglu32 -lopengl32 -lOpenCL
CLEAN=rm *.exe *.o *.a
else
#  OSX
ifeq "$(shell uname)" "Darwin"
CFLG=-O3 -Wall -Wno-deprecated-declarations -DRES=2
LIBS=-framework GLUT -framework OpenGL -framework OpenCL
#  Linux/Unix/Solaris
else
CFLG=-O3 -Wall
LIBS=-lglut -lGLU -lGL -lm -lOpenCL
endif
#  OSX/Linux/Unix/Solaris
CLEAN=rm -f final *.o *.a
endif

#  Compile and link
final:final.c
	gcc $(CFLG) -o $@ $^   $(LIBS)

#  Clean
clean:
	$(CLEAN)
