#!/usr/bin/python
"""
Extract data from UK Met Office Rain Radar NIMROD image files.

Parse NIMROD format image files, display header data and allow extraction of
raster image to an ESRI ASCII (.asc) format file. A bounding box may be
specified to clip the image to the area of interest. Can be imported as a
Python module or run directly as a command line script.

Author: Richard Thomas
Version: 1.0 (13 April 2015)
Public Repository: https://github.com/richard-thomas/MetOffice_NIMROD

Command line usage:
  python nimrod.py [-h] [-q] [-x] [-bbox XMIN XMAX YMIN YMAX] [infile] [outfile]

positional arguments:
  infile                (Uncompressed) NIMROD input filename
  outfile               Output raster filename (*.asc)

optional arguments:
  -h, --help            show this help message and exit
  -q, --query           Display metadata
  -x, --extract         Extract raster file in ASC format
  -bbox XMIN XMAX YMIN YMAX
                        Bounding box to clip raster data to

Note that any bounding box must be specified in the same units and projection
as the input file. The bounding box does not need to be contained by the input
raster but must intersect it.

Example command line usage:
  python nimrod.py -bbox 279906 285444 283130 290440
    -xq 200802252000_nimrod_ng_radar_rainrate_composite_1km_merged_UK_zip
    plynlimon_catchments_rainfall.asc

Example Python module usage:
    import nimrod
    a = nimrod.Nimrod(open(
        '200802252000_nimrod_ng_radar_rainrate_composite_1km_merged_UK_zip'))
    a.query()
    a.extract_asc(open('full_raster.asc', 'w'))
    a.apply_bbox(279906, 285444, 283130, 290440)
    a.query()
    a.extract_asc(open('clipped_raster.asc', 'w'))

Notes:
  1. Valid for v1.7 and v2.6-4 of NIMROD file specification
  2. Assumes image origin is top left (i.e. that header[24] = 0)
  3. Tested on UK composite 1km and 5km data, under Linux and Windows XP
  4. Further details of NIMROD data and software at the NERC BADC website:
      http://badc.nerc.ac.uk/browse/badc/ukmo-nimrod/   

Copyright (c) 2015 Richard Thomas
(Nimrod.__init__() method based on read_nimrod.py by Charles Kilburn Aug 2008)

This program is free software: you can redistribute it and/or modify
it under the terms of the Artistic License 2.0 as published by the
Open Source Initiative (http://opensource.org/licenses/Artistic-2.0)

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
"""

import sys
import struct
import array

class Nimrod:
    """Reading, querying and processing of NIMROD format rainfall data files."""

    class RecordLenError(Exception):
        """
        Exception Type: NIMROD record length read from file not as expected.
        """
    
        def __init__(self, actual, expected, location):
            self.message = (
                "Incorrect record length %d bytes (expected %d) at %s."
                % (actual, expected, location))

    class HeaderReadError(Exception):
        """Exception Type: Read error whilst parsing NIMROD header elements."""
        pass
 
    class PayloadReadError(Exception):
        """Exception Type: Read error whilst parsing NIMROD raster data."""
        pass
 
    class BboxRangeError(Exception):
        """
        Exception Type: Bounding box specified out of range of raster image.
        """
        pass
 
 
    def __init__(self, infile):
        """
        Parse all header and data info from a NIMROD data file into this object.
        (This method based on read_nimrod.py by Charles Kilburn Aug 2008)
                
        Args:
            infile: NIMROD file object opened for binary reading
        Raises:
            RecordLenError: NIMROD record length read from file not as expected
            HeaderReadError: Read error whilst parsing NIMROD header elements
            PayloadReadError: Read error whilst parsing NIMROD raster data
        """
        
        def check_record_len(infile, expected, location):
            """
            Check record length in C struct is as expected.
            
            Args:
                infile: file to read from
                expected: expected value of record length read
                location: description of position in file (for reporting)
            Raises:
                HeaderReadError: Read error whilst reading record length
                RecordLenError: Unexpected NIMROD record length read from file
            """
            
            # Unpack length from C struct (Big Endian, 4-byte long)
            try:
                record_length, = struct.unpack(">l", infile.read(4))
            except Exception:
                raise Nimrod.HeaderReadError
            if record_length != expected:
                raise Nimrod.RecordLenError(record_length, expected, location)
        
        
        # Header should always be a fixed length record
        check_record_len(infile, 512, "header start")
        
        try:
            # Read first 31 2-byte integers (header fields 1-31)
            gen_ints = array.array("h")
            gen_ints.fromfile(infile, 31)
            gen_ints.byteswap()
            
            # Read next 28 4-byte floats (header fields 32-59)
            gen_reals = array.array("f")
            gen_reals.fromfile(infile, 28)
            gen_reals.byteswap()
            
            # Read next 45 4-byte floats (header fields 60-104)
            spec_reals = array.array("f")
            spec_reals.fromfile(infile, 45)
            spec_reals.byteswap()
            
            # Read next 56 characters (header fields 105-107)
            characters = array.array("b")
            characters.fromfile(infile, 56)
            
            # Read next 51 2-byte integers (header fields 108-)
            spec_ints = array.array("h")
            spec_ints.fromfile(infile, 51)
            spec_ints.byteswap()
        except Exception:
            infile.close()
            raise Nimrod.HeaderReadError
    
        check_record_len(infile, 512, "header end")

        # Extract strings and make duplicate entries to give meaningful names
        chars = characters.tobytes()
        self.units = chars[0:8]
        self.data_source = chars[8:32]
        self.title = chars[32:55]

        # Store header values in a list so they can be indexed by "element
        # number" shown in NIMROD specification (starts at 1)
        self.hdr_element = [None]           # Dummy value at element 0
        self.hdr_element.extend(gen_ints)
        self.hdr_element.extend(gen_reals)
        self.hdr_element.extend(spec_reals)
        self.hdr_element.extend([self.units])
        self.hdr_element.extend([self.data_source])
        self.hdr_element.extend([self.title])
        self.hdr_element.extend(spec_ints)
        
        # Duplicate some of values to give more meaningful names
        self.nrows = self.hdr_element[16]
        self.ncols = self.hdr_element[17]
        self.n_data_specific_reals = self.hdr_element[22]
        self.n_data_specific_ints = self.hdr_element[23] + 1
            # Note "+ 1" because header value is count from element 109nimrod_to_kmz
        self.y_top = self.hdr_element[34]
        self.y_pixel_size = self.hdr_element[35]
        self.x_left = self.hdr_element[36]
        self.x_pixel_size = self.hdr_element[37]

        # Calculate other image bounds (note these are pixel centres)
        self.x_right = (self.x_left + self.x_pixel_size * (self.ncols - 1))
        self.y_bottom = (self.y_top - self.y_pixel_size * (self.nrows - 1))

        # Read payload (actual raster data)
        array_size = self.ncols * self.nrows
        check_record_len(infile, array_size * 2, "data start")
             
        self.data = array.array("h")
        try:
            self.data.fromfile(infile, array_size)
            self.data.byteswap()
        except Exception:
            infile.close()
            raise Nimrod.PayloadReadError

        check_record_len(infile, array_size * 2, "data end")
        infile.close()

    
    def query(self):
        """Print complete NIMROD file header information."""
        
        print("NIMROD file raw header fields listed by element number:")
        print("General (Integer) header entries:")
        for i in range(1, 32):
            print(" ", i, "\t", self.hdr_element[i])
        print("General (Real) header entries:")
        for i in range(32, 60):
            print(" ", i, "\t", self.hdr_element[i])
        print ("Data Specific (Real) header entries (%d):"
               % self.n_data_specific_reals)
        for i in range(60, 60 + self.n_data_specific_reals):
            print(" ", i, "\t", self.hdr_element[i])
        print ("Data Specific (Integer) header entries (%d):"
               % self.n_data_specific_ints)
        for i in range(108, 108 + self.n_data_specific_ints):
            print(" ", i, "\t", self.hdr_element[i])
        print("Character header entries:")
        print("  105 Units:           ", self.units)
        print("  106 Data source:     ", self.data_source)
        print("  107 Title of field:  ", self.title)
            
        # Print out info & header fields
        # Note that ranges are given to the edge of each pixel
        print("\nValidity Time:  %2.2d:%2.2d on %2.2d/%2.2d/%4.4d" % (
            self.hdr_element[4], self.hdr_element[5],
            self.hdr_element[3], self.hdr_element[2], self.hdr_element[1]))
        print ("Easting range:  %.1f - %.1f (at pixel steps of %.1f)"
               % (self.x_left - self.x_pixel_size / 2,
                  self.x_right + self.x_pixel_size / 2, self.x_pixel_size))
        print ("Northing range: %.1f - %.1f (at pixel steps of %.1f)"
               % (self.y_bottom - self.y_pixel_size / 2,
                  self.y_top + self.y_pixel_size / 2, self.y_pixel_size))
        print("Image size: %d rows x %d cols" % (self.nrows, self.ncols))

   
    def apply_bbox(self, xmin, xmax, ymin, ymax):
        """
        Clip raster data to all pixels that intersect specified bounding box.

        Note that existing object data is modified and all header values
        affected are appropriately adjusted. Because pixels are specified by
        their centre points, a bounding box that comes within half a pixel
        width of the raster edge will intersect with the pixel.
        
        Args:
            xmin: Most negative easting or longitude of bounding box
            xmax: Most positive easting or longitude of bounding box
            ymin: Most negative northing or latitude of bounding box
            ymax: Most positive northing or latitude of bounding box
        Raises:
            BboxRangeError: Bounding box specified out of range of raster image
        """
        
        # Check if there is no overlap of bounding box with raster
        if (
                xmin > self.x_right  + self.x_pixel_size / 2 or
                xmax < self.x_left   - self.x_pixel_size / 2 or
                ymin > self.y_top    + self.y_pixel_size / 2 or
                ymax < self.y_bottom - self.x_pixel_size / 2):
            raise Nimrod.BboxRangeError

        # Limit bounds to within raster image
        xmin = max(xmin, self.x_left)
        xmax = min(xmax, self.x_right)
        ymin = max(ymin, self.y_bottom)
        ymax = min(ymax, self.y_top)

        # Calculate min and max pixel index in each row and column to use
        # Note addition of 0.5 as x_left location is centre of pixel
        # ('int' truncates floats towards zero)
        xMinPixelId = int((xmin - self.x_left) / self.x_pixel_size + 0.5)
        xMaxPixelId = int((xmax - self.x_left) / self.x_pixel_size + 0.5)
        
        # For y (northings), note the first data row stored is most north 
        yMinPixelId = int((self.y_top - ymax) / self.y_pixel_size + 0.5)
        yMaxPixelId = int((self.y_top - ymin) / self.y_pixel_size + 0.5)
          
        bbox_data = []
        for i in range(yMinPixelId, yMaxPixelId + 1):
            bbox_data.extend(self.data[i * self.ncols + xMinPixelId:
                                       i * self.ncols + xMaxPixelId + 1])
            
        # Update object where necessary
        self.data = bbox_data
        self.x_right = self.x_left + xMaxPixelId * self.x_pixel_size
        self.x_left += xMinPixelId * self.x_pixel_size
        self.ncols = xMaxPixelId - xMinPixelId + 1
        self.y_bottom = self.y_top - yMaxPixelId * self.y_pixel_size
        self.y_top -= yMinPixelId * self.y_pixel_size
        self.nrows = yMaxPixelId - yMinPixelId + 1
        self.hdr_element[16] = self.nrows
        self.hdr_element[17] = self.ncols
        self.hdr_element[34] = self.y_top
        self.hdr_element[36] = self.x_left


    def extract_asc (self, outfile):
        """
        Write raster data to an ESRI ASCII (.asc) format file.
        
        Args:
            outfile: file object opened for writing text
        """
        
        # As ESRI ASCII format only supports square pixels, warn if not so
        if self.x_pixel_size != self.y_pixel_size:
            print ("Warning: x_pixel_size(%d) != y_pixel_size(%d)"
                   % (self.x_pixel_size, self.y_pixel_size))
                
        # Write header to output file. Note that data is valid at the centre
        # of each pixel so "xllcenter" rather than "xllcorner" must be used  
        outfile.write("ncols          %d\n" % self.ncols)
        outfile.write("nrows          %d\n" % self.nrows)
        outfile.write("xllcenter     %d\n" % self.x_left)
        outfile.write("yllcenter     %d\n" % self.y_bottom)
        outfile.write("cellsize       %.1f\n" % self.y_pixel_size)
        outfile.write("nodata_value  %.1f\n" % self.hdr_element[38])
    
        # Write raster data to output file    
        for i in range(self.nrows):
            for j in range(self.ncols - 1):
                outfile.write("%d " % self.data[i*self.ncols + j])
            outfile.write("%d\n" % self.data[i*self.ncols + self.ncols - 1])
        outfile.close()


#-------------------------------------------------------------------------------
# Handle if called as a command line script
# (And as an example of how to invoke class methods from an importing module)
#-------------------------------------------------------------------------------
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract information and data from a NIMROD format file",
        epilog="""Note that any bounding box must be specified in the same
                  units and projection as the input file. The bounding box
                  does not need to be contained by the input raster but
                  must intersect it.""")
    parser.add_argument("-q", "--query", action="store_true",
                        help="Display metadata")
    parser.add_argument("-x", "--extract", action="store_true",
                        help="Extract raster file in ASC format")
    parser.add_argument('infile', nargs='?', type=argparse.FileType('rb'),
                        default=sys.stdin,
                        help="(Uncompressed) NIMROD input filename")
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'),
                        default=sys.stdout,
                        help="Output raster filename (*.asc)")
    parser.add_argument("-bbox", type=float, nargs=4,
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help="Bounding box to clip raster data to")
    args = parser.parse_args()

    if not args.query and not args.extract:
        parser.print_help()
        sys.exit(1)
        
    # Initialise data object by reading NIMROD file
    # (Only trap record length exception as others self-explanatory)
    try:
        rainfall_data = Nimrod(args.infile)
    except Nimrod.RecordLenError as error:
        sys.stderr.write("ERROR: %s\n" % error.message)
        sys.exit(1)
          
    if args.bbox:
        sys.stderr.write(
            "Trimming NIMROD raster to bounding box...\n")
        try:
            rainfall_data.apply_bbox(*args.bbox)
        except Nimrod.BboxRangeError:
            sys.stderr.write("ERROR: bounding box not within raster image.\n")
            sys.exit(1)

    # Perform query after any bounding box trimming to allow sanity checking of
    # size of resulting image
    if args.query:
        rainfall_data.query()
        
    if args.extract:
        sys.stderr.write(
            "Extracting NIMROD raster to ASC file...\n")
        sys.stderr.write(
            "  Outputting data array (%d rows x %d cols = %d pixels)\n"
            % (rainfall_data.nrows, rainfall_data.ncols,
               rainfall_data.nrows * rainfall_data.ncols))
        rainfall_data.extract_asc(args.outfile)
    
