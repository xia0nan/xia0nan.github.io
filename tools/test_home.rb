# frozen_string_literal: true

require "tmpdir"
require "fileutils"
require "open3"
require "nokogiri"

root = File.expand_path("..", __dir__)

def check(condition, message)
  abort "FAIL: #{message}" unless condition
end

def build(source, output, root)
  log, status = Open3.capture2e({ "JEKYLL_ENV" => "production" }, "bundle", "exec", "jekyll", "build", "--source", source, "--destination", output, chdir: root)
  abort log unless status.success?
  check(!log.include?("Conflict:"), "no output collisions")
end

def previews(path)
  Nokogiri::HTML5(File.read(path)).css("#post-list a.post-preview").map { |a| a["href"] }
end

Dir.mktmpdir("nanx-home-test-") do |temp|
  source = File.join(temp, "source")
  output = File.join(temp, "output")
  FileUtils.mkdir_p(source)
  excluded = %w[.git .vscode .bundle vendor _site .jekyll-cache .jekyll-metadata]
  Dir.children(root).reject { |name| excluded.include?(name) }.each do |name|
    FileUtils.cp_r(File.join(root, name), source)
  end
  # Isolate fixtures from later real articles while retaining historical posts.
  Dir.glob(File.join(source, "_posts", "*.md")).each do |path|
    File.delete(path) unless File.read(path).include?("archived: true")
  end
  build(source, output, root)
  home = Nokogiri::HTML5(File.read(File.join(output, "index.html")))
  check(home.at_css("#intro-title"), "zero-post homepage retains introduction")
  check(home.css("#post-list, #access-lastmod, nav.pagination").empty?, "zero-post homepage has no empty writing panels")
  check(!File.exist?(File.join(output, "page")), "no empty pagination pages")

  (1..12).each do |number|
    post = "---\nlayout: post\ntitle: Fixture #{number}\npin: #{number <= 3}\n---\nTemporary validation content.\n"
    File.write(File.join(source, "_posts", "2026-01-#{format('%02d', number)}-fixture-#{number}.md"), post)
  end
  # Even an accidentally pinned archive must stay off the homepage.
  archived = Dir.glob(File.join(source, "_posts", "*.md")).find { |p| File.read(p).include?("archived: true") }
  File.write(archived, File.read(archived).sub("archived: true", "archived: true\npin: true"))
  build(source, output, root)
  expected = [3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4].map { |n| "/posts/fixture-#{n}/" }
  check(previews(File.join(output, "index.html")) == expected.first(10), "pinned articles first, no archived cards")
  check(previews(File.join(output, "page/2/index.html")) == expected.last(2), "pagination has no duplicates or omissions")
  page_two = Nokogiri::HTML5(File.read(File.join(output, "page/2/index.html")))
  check(!page_two.at_css("#intro-title"), "introduction only on first page")
  check(page_two.at_css('a[aria-label="previous-page"]')&.[]("href") == "/", "pagination returns home")
  check(File.read(File.join(output, "page2/index.html")).include?("https://nanx.cc/archives/"), "legacy redirect survives new pagination")
end
puts "PASS: zero-post homepage, pinned posts, archive isolation and future pagination"
