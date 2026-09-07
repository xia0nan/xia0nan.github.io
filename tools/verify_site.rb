# frozen_string_literal: true

require "json"
require "digest"
require "nokogiri"
require "yaml"
require "date"

ROOT = File.expand_path("..", __dir__)
OUTPUT = File.join(ROOT, "_site")
BASELINE = JSON.parse(File.read(File.join(__dir__, "archive-baseline.json")))

def check(condition, message)
  abort "FAIL: #{message}" unless condition
end

def html(path)
  Nokogiri::HTML5(File.read(File.join(OUTPUT, path)))
end

home = html("index.html")
archive = html("archives/index.html")
search = JSON.parse(File.read(File.join(OUTPUT, "assets/js/data/search.json")))
feed = Nokogiri::XML(File.read(File.join(OUTPUT, "atom.xml"))) { |c| c.strict }
ns = { "a" => "http://www.w3.org/2005/Atom" }
sitemap = Nokogiri::XML(File.read(File.join(OUTPUT, "sitemap.xml"))) { |c| c.strict }
config = YAML.safe_load(File.read(File.join(ROOT, "_config.yml")))
check(config["url"] == "https://nanx.cc" && config["baseurl"] == "", "custom-domain configuration")
check(File.read(File.join(OUTPUT, "CNAME")).strip == "nanx.cc", "custom domain preserved")
check(home.at_css("#intro-title")&.text&.include?("production AI systems"), "professional introduction")
check(home.at_css('link[rel="canonical"]')&.[]("href") == "https://nanx.cc/", "home canonical URL")
check(home.css("#sidebar nav a").map { |a| a["href"] } == ["/", "/about/", "/archives/"], "launch navigation")

BASELINE.each do |post|
  source = File.read(File.join(ROOT, post.fetch("source")))
  front, body = source.delete_prefix("---\n").split("\n---", 2)
  metadata = YAML.safe_load(front, permitted_classes: [Date, Time])
  check(Digest::SHA256.hexdigest(body) == post.fetch("body_sha256"), "unchanged body: #{post['source']}")
  check(metadata["hidden"] && metadata["archived"], "archive flags: #{post['source']}")
  check(metadata["permalink"] == post["url"], "permalink preserved: #{post['source']}")
  page = html(post.fetch("url").delete_prefix("/"))
  check(page.at_css("article h1")&.text == post["title"], "article title: #{post['url']}")
  check(page.at_css(".archive-notice"), "historical notice: #{post['url']}")
  check(page.at_css('link[rel="canonical"]')&.[]("href") == "https://nanx.cc#{post['url']}", "article canonical: #{post['url']}")
  check(archive.css("#archives a").any? { |a| a["href"] == post["url"] }, "archive includes #{post['url']}")
  check(search.any? { |item| item["url"] == post["url"] }, "search includes #{post['url']}")
  check(home.css("#post-list a, #access-lastmod a").none? { |a| a["href"] == post["url"] }, "archive absent from current writing")
  entry = feed.xpath("//a:entry", ns).find { |node| node.at_xpath("a:link", ns)["href"] == "https://nanx.cc#{post['url']}" }
  check(entry && entry.at_xpath("a:published", ns).text.start_with?(post["date"]), "feed preserves publication date")
  check(sitemap.to_xml.include?("https://nanx.cc#{post['url']}"), "sitemap includes archive")
end

[["about.html", "/about/"], ["page2/index.html", "/archives/"]].each do |file, target|
  page = html(file)
  check(page.at_css('meta[http-equiv="refresh"]')&.[]("content")&.include?(target), "redirect #{file}")
  check(page.at_css('link[rel="canonical"]')&.[]("href") == "https://nanx.cc#{target}", "redirect canonical #{file}")
end
check(html("about/index.html").at_css("main").text.include?("Shawn Xiao"), "About identity")
check(html("404.html").at_css("main a[href='/archives/']"), "useful 404")
check(File.file?(File.join(OUTPUT, "images/2020-02-26-final-compare.png")), "historical image")

%w[docs tools _drafts vendor README.md Gemfile Gemfile.lock].each do |path|
  check(!File.exist?(File.join(OUTPUT, path)), "excluded output: #{path}")
end
forbidden = ["Spring Overload", "2021 Half Year Review", "From ML Models to AI Agents: How the Production Stack Is Changing", "LLM Evaluation Is a System, Not a Metric"]
Dir.glob(File.join(OUTPUT, "**", "*.{html,xml,json}")).each do |path|
  text = File.read(path)
  forbidden.each { |phrase| check(!text.include?(phrase), "unpublished content in #{path}") }
  next unless path.end_with?(".html")
  check(!text.include?("CC BY 4.0"), "no accidental article relicensing")
  check(!text.include?("app.min.js?"), "PWA disabled")
end
puts "PASS: archive bodies, URLs, search, feed, redirects, metadata and publication boundaries"
