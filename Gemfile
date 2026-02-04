source "https://rubygems.org"

gem "jekyll", "~> 3.9"
gem "jekyll-feed", "~> 0.15"
gem "kramdown-parser-gfm"

# Fix ffi version for Ruby 2.6 compatibility
gem "ffi", "~> 1.15.5"

# Windows and JRuby does not include zoneinfo files, so bundle the tzinfo-data gem
# and associated library.
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance and livereload
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]
