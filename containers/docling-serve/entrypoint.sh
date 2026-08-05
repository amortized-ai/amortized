#!/bin/sh
chonkie serve --port 8900 &
exec docling-serve serve --host 0.0.0.0 --port 5001
