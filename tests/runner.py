import coverage

cov = coverage.Coverage()
cov.start()

from tests import test_2d
from tests import test_3d
from tests import test_3d_collider
from tests import test_3d_mesh

cov.stop()
cov.xml_report()
cov.html_report()
