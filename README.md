# NLP Machine Translation Test

A broadly labeled Bible translation project for developing and experimenting
with NLP related machine translation tools and libraries.

## Status

**Private Development & Invite Only**

To avoid conflicts and confusion with other projects, this testing project is
privately shared.  Locally modified libraries within this project should also
be kept private until vetted code can be officially released via a pull request
or other formal method of submission.

## Project Directory Structure

* `data/` - Individual small files (<100MB) of shared data to be used by the
rest of the project.  Do not forget to include an extension in the name to help
tract data types. Larger files or sets of data should be kept local outside of
the project directory, or utilize other cloud storage (like AWS S3). 
* `docs/` - All document-style file types, including Python notebooks, various
types of markdown, HTML output, etc.
* `Python/` - Python script files used by project are sourced from here.
* `R/` - R script files used by project are sourced from here.
* `vendors/` - Empty placeholder directory to store locally any third-party
libraries, drivers, and other tools that will not be copied to the project
repository.
 