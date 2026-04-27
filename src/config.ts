export const SITE = {
  website: "https://chaitanya-19.github.io/",
  author: "Chaitanya Kulkarni",
  profile: "https://github.com/chaitanya-19",
  desc: "Modern LLM architecture, from vanilla GPT-2 to frontier models. A technical blog series for engineers who want to understand LLM internals at parameter-level depth.",
  title: "Chaitanya Kulkarni",
  ogImage: "astropaper-og.jpg",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true,
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/chaitanya-19/chaitanya-19.github.io/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr",
  lang: "en",
  timezone: "America/Los_Angeles",
} as const;