```{=html}
<div class="student-project-grid">
<% for (const item of items) {
  if (!Array.isArray(item.students) || !item.students.length || item.students.some(s => typeof s !== 'string' || !s.trim())) throw new Error('Project students must be a nonempty list of names: ' + item.title);
  const status = item.status || 'Proposal approved';
  if (!['Proposal approved', 'In progress', 'Completed'].includes(status)) throw new Error('Unknown project status: ' + status);
  const thumbnail = item.thumbnail || 'images/placeholder.svg';
  if (!/^images\/[a-zA-Z0-9_./-]+\.(svg|png|jpe?g|webp)$/i.test(thumbnail) || thumbnail.includes('..')) throw new Error('Use a local images/ thumbnail: ' + item.title);
  if (item.thumbnail && !item.thumbnail_alt) throw new Error('Add thumbnail_alt for ' + item.title);
%>
<article class="student-project-card">
  <figure class="student-project-visual">
    <img src="<%- thumbnail %>" alt="<%- item.thumbnail_alt || 'Abstract network illustration; project thumbnail forthcoming.' %>" width="1200" height="750" loading="lazy">
    <% if (item.thumbnail_credit) { %><figcaption><%- item.thumbnail_credit %></figcaption><% } %>
  </figure>
  <div class="student-project-copy">
    <span class="student-project-status"><%- item.example ? 'Fictional example' : status %></span>
    <h3><%- item.title %></h3>
    <p class="student-project-team"><%- item.students.join(' · ') %></p>
    <p class="student-project-description"><%- item.description %></p>
    <% if (item.example) { %><p class="student-project-example">Sample proposal with fictional names. This is a layout preview, not a student submission.</p><% } %>
    <% const links = ['report', 'code', 'demo'].filter(key => item[key]); if (links.length) { %>
    <nav class="student-project-links" aria-label="Resources for <%- item.title %>">
      <% for (const key of links) { if (!/^https:\/\//i.test(item[key])) throw new Error('Project links must use https://'); %>
      <a href="<%- item[key] %>"><%- {report: 'Read report', code: 'View code', demo: 'Open demo'}[key] %> ↗</a>
      <% } %>
    </nav>
    <% } %>
  </div>
</article>
<% } %>
</div>
<% if (!items.length) { %><p>Approved projects will appear here. Start with the proposal template below.</p><% } %>
```
