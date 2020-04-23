import coverage

cov = coverage.Coverage()
cov.start()

import test_2d
import test_3d
import test_3d_collider
import test_3d_mesh

cov.stop()
cov.xml_report()
cov.html_report()
