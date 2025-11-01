import fs from "fs";
import path from "path";
import { fileURLToPath } from "url"; // 新增这一行

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const getTutorialSidebar = () => {
  const dir = path.resolve(__dirname, "../tutorials"); // 现在可以正常使用 __dirname 了
  const files = fs.readdirSync(dir).filter((f: string) => f.endsWith(".md"));
  return [
    {
      text: "教程列表",
      items: files.map((file: string) => ({
        text: file.replace(".md", ""),
        link: `/tutorials/${file.replace(".md", "")}`,
      })),
    },
  ];
};

export default {
  base: "/",
  title: "我的 Markdown 文章",
  description: "用 VitePress 搭建的文章展示网站",
  // 记得根据部署路径配置 base（重要！）
  // base: '/你的仓库名/',  // 部署到 GitHub Pages 时需要

  themeConfig: {
    nav: [
      { text: "首页", link: "/" },
      {
        text: "教程",
        items: [
          { text: "教程1", link: "../tutorials/first-tutorial" },
          { text: "教程2", link: "../tutorials/second-tutorial" },
        ],
      },
      {
        text: "笔记",
        items: [{ text: "学习笔记", link: "../notes/study-note" }],
      },
      { text: "第二篇文章", link: "/second-article" },
      { text: "关于", link: "/about" },
    ],

    sidebar: {
      "/tutorials/": getTutorialSidebar(), // 动态生成教程侧边栏
      "/notes/": [
        {
          text: "笔记列表",
          items: [{ text: "学习笔记", link: "../notes/study-note" }],
        },
      ],
      "/": [
        {
          text: "首页文章",
          items: [
            { text: "首页", link: "/" },
            { text: "第二篇文章", link: "/second-article" },
          ],
        },
      ],
    },

    footer: {
      message: "部署于 GitHub Pages | 欢迎访问",
    },
  },
};
