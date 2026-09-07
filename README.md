# nanx.cc

Shawn Xiao’s technical blog, built with Jekyll and the pinned Chirpy 7.6.0 gem. GitHub Pages serves the site at <https://nanx.cc>.

## Local preview

Use Ruby 3.4.10 (see `.ruby-version`). On macOS with Homebrew:

```sh
brew install ruby@3.4
export PATH="/opt/homebrew/opt/ruby@3.4/bin:$PATH"
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve
```

Open <http://127.0.0.1:4000>. Jekyll reloads content changes; restart the server after changing `_config.yml`. Dependencies and build output are ignored by Git. Intel Macs should use their Homebrew installation’s Ruby path.

## Validation

```sh
JEKYLL_ENV=production bundle exec jekyll build
bundle exec ruby tools/verify_site.rb
bundle exec ruby tools/test_home.rb
bundle exec htmlproofer _site --disable-external --no-enforce-https
```

The checks verify the six historical article bodies against the pre-migration baseline, URLs, archive/search/feed visibility, canonical metadata, redirects, and exclusion of unpublished material. Temporary fixtures test an empty homepage, pinned posts and two-page pagination without adding content to the real site.

External link checking and HTTPS enforcement for outbound links are disabled because historical articles retain their original HTTP references. Internal links and fragment targets remain checked. The website itself uses HTTPS.

## Publishing

The `Build and Deploy` GitHub Actions workflow validates pull requests and deploys successful builds from `master`. GitHub Pages must use **GitHub Actions** as its build source. Keep the `CNAME` file and the custom domain setting at `nanx.cc`, with HTTPS enforced.

Before deployment, all build and verification steps must pass. After deployment, check Home, About, Archives, a historical article, `/atom.xml`, and the compatibility redirects. For rollback, revert the relaunch commit and restore the previous Pages build source (`master`, repository root, legacy build); do not reset repository history.

## Writing next

Keep proposed articles in [the content roadmap](docs/content-roadmap.md) until they are ready. `docs/`, `_drafts/`, and `tools/` do not ship in the generated website. The reference remains visible in this public source repository.

Publish an article as `_posts/YYYY-MM-DD-slug.md`:

```yaml
---
layout: post
title: Your finished article title
categories: [AI Systems]
tags: [agents, system-design]
# pin: true
# math: true
# mermaid: true
# last_modified_at: YYYY-MM-DD
---
```

New posts automatically appear on Home and in the feed, search and archives. Pin up to three cornerstone articles once published. Set `last_modified_at` explicitly only for substantive content updates; theme changes should not make articles look newly edited. Comments, analytics and PWA are disabled.

Categories and Tags navigation, a curated Projects page, and new articles are deferred. The category/tag generators are available for future writing. Add navigation tabs only when useful content exists.

## Historical archives

The six original articles remain in `_posts` with `archived: true`, `hidden: true`, and fixed original permalinks. Keep both flags together. `hidden` removes them from homepage pagination; `archived` adds the historical notice and removes them from Recently Updated. Their bodies, dates, titles, images and heading anchors are preserved. Do not remove the fixed permalinks when adding categories.

Compatibility routes:

- Original dated `.html` article URLs continue to serve the articles directly.
- `/about.html` redirects to `/about/`.
- `/page2/` redirects to `/archives/`; new homepage pagination uses `/page/:num/`.
- `/atom.xml` retains the original feed endpoint and article entry IDs.

The migration baseline in `tools/archive-baseline.json` protects the original bodies. If an intentional historical correction is needed later, review it explicitly before updating that baseline.

## Theme maintenance

Most UI comes from the gem. Local overrides provide the introduction and empty-home behavior, archive notice, filtered Recently Updated panel, accessible viewport metadata, existing favicon links, feed discovery, and copyright wording. Review those overrides against upstream when upgrading the pinned theme. Styles retain Chirpy’s standard light/dark appearance.

The original Hyde license remains in `LICENSE.md`. See `THIRD_PARTY_NOTICES.md` for the license covering adapted Chirpy templates. Theme migration does not grant a new license to the articles.
