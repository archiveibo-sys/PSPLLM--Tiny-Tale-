TARGET = tinytale
OBJS = main.o

INCDIR =
CFLAGS = -O3 -G0 -Wall -ffast-math -fsingle-precision-constant
CXXFLAGS = $(CFLAGS) -fno-exceptions -fno-rtti
ASFLAGS = $(CFLAGS)

LIBDIR =
LDFLAGS =
LIBS = -lpspdebug -lpspdisplay -lpspge -lpspctrl -lpsppower -lm

EXTRA_TARGETS = EBOOT.PBP
PSP_EBOOT_TITLE = PSP TinyTale LLM

PSPSDK = $(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak
