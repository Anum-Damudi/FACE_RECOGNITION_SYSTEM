Sample criminal dataset layout (copy folders into `criminals/` and add face images):

criminals/
  JANE_DOE_001/
    metadata.json
    reference_01.jpg
  JOHN_ROE_002/
    metadata.json
    mugshot.png

Each metadata.json must include at least display_name and crime_type.
Supported image extensions: .jpg .jpeg .png .bmp .webp

After adding images, POST /api/v1/reload-gallery or restart the server.
