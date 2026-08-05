#!/bin/sh
chonkie serve --port 8900 &
exec docling-serve run
