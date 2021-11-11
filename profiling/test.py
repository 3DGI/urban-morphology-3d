import cityStats
from click.testing import CliRunner

# This runs on gilfoyle
if __name__ == '__main__':
    runner = CliRunner()
    result = runner.invoke(cityStats.main,
                           args=["-b",
                                 "-dsn", "dbname=baseregisters",
                                 "-o", "/home/bdukai/software/urban-morphology-3d/tmp/4434_lod2_surface_areas.csv",
                                 "--",
                                 "/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_4434.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_4435.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_4432.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_7296.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_4433.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_4440.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_4436.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_4439.json","/data/3DBAGv2/export/cityjson/v210908_fd2cee53/3dbag_v210908_fd2cee53_1720.json"])
