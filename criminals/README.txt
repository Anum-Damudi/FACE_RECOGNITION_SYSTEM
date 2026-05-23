Copy folders from ../sample_gallery/ into this directory (or create your own),
add face photographs next to each metadata.json, then call POST /api/v1/reload-gallery.

The API loads only subfolders that contain metadata.json and at least one image file.
