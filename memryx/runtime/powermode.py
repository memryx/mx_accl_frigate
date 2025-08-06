import os


def get_power_tuple():

    DEFAULT_FREQ=600

    conffile="/etc/memryx/power.conf"

    freq4c = 0
    volt4c = 0
    freq2c = 0
    volt2c = 0

    if os.path.isfile(conffile):

        try:
            with open(conffile, "r") as f:
                lines = [line.rstrip() for line in f]

                for line in lines:
                    if line.startswith("#"):
                        pass
                    else:
                        var = line.split("=")[0]
                        val = line.split("=")[1]
                        if var == "FREQ4C":
                            freq4c = int(val)
                        elif var == "VOLT4C":
                            volt4c = int(val)
                        elif var == "FREQ2C":
                            freq2c = int(val)
                        elif var == "VOLT2C":
                            volt2c = int(val)

                # enforce limits
                if freq4c < 200:
                    freq4c = 200
                elif freq4c > 850:
                    freq4c = 850

                if volt4c < 680:
                    volt4c = 680
                elif volt4c > 800:
                    volt4c = 800


                if freq2c < 200:
                    freq2c = 200
                elif freq2c > 850:
                    freq2c = 850

                if volt2c < 680:
                    volt2c = 680
                elif volt2c > 800:
                    volt2c = 800
        except:
            freq4c = DEFAULT_FREQ
            volt4c = 700
            freq2c = DEFAULT_FREQ
            volt2c = 700

    else:

        freq4c = DEFAULT_FREQ
        volt4c = 700
        freq2c = DEFAULT_FREQ
        volt2c = 700

    return (freq4c, volt4c, freq2c, volt2c)
