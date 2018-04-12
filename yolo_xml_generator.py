import os
from jinja2 import Environment, PackageLoader


class xml_fill:
    def __init__(self, path, width, height, depth=3, database='Unknown', segmented=0):
        environment = Environment(loader=PackageLoader('convert_yolo_xml.source', 'XML_template'), keep_trailing_newline=True)
        self.annotation_template = environment.get_template('yolo_template.xml')

        abspath = os.path.abspath(path)

        self.template_parameters = {
            'path': abspath,
            'filename': os.path.basename(abspath),
            'folder': os.path.basename(os.path.dirname(abspath)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': database,
            'segmented': segmented,
            'objects': []
        }

    def addBox(self, name, x, y, w, h, pose='Unspecified', truncated=0, difficult=0):
        self.template_parameters['objects'].append({
            'name': name,
            'xmin': x,
            'ymin': y,
            'xmax': x + w,
            'ymax': y + h,
            'pose': pose,
            'truncated': truncated,
            'difficult': difficult,
        })

    def save(self, annotation_path):
        with open(annotation_path, 'w') as file:
            content = self.annotation_template.render(**self.template_parameters)
            file.write(content)