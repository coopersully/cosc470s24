import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;

public class MSPaintButWorse extends JFrame {
    private static final int WIDTH = 28;
    private static final int HEIGHT = 28;
    private final BufferedImage image;

    public MSPaintButWorse() {
        super("It's like MS Paint... But Worse!");
        initUI();
        this.image = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
    }

    private void initUI() {
        JPanel canvas = createCanvas();
        add(canvas, BorderLayout.CENTER);
        setupMenuBar();
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private JPanel createCanvas() {
        JPanel canvas = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                super.paintComponent(g);
                g.drawImage(image, 0, 0, this.getWidth(), this.getHeight(), null);
            }
        };
        canvas.setPreferredSize(new Dimension(WIDTH, HEIGHT));
        canvas.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                drawPixel(e.getX(), e.getY());
            }
        });
        return canvas;
    }

    private void drawPixel(int x, int y) {
        if (x < WIDTH && y < HEIGHT) { // Prevent drawing outside the bounds
            Graphics2D g = image.createGraphics();
            g.setColor(Color.WHITE);
            g.fillRect(x, y, 1, 1);
            g.dispose();
            repaint();
        }
    }

    private void saveImage(String filename) {
        try {
            if (ImageIO.write(image, "png", new File(filename))) {
                System.out.println("Image saved as " + filename);
            } else {
                System.out.println("Failed to save the image.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void setupMenuBar() {
        JMenuBar menuBar = new JMenuBar();
        JMenu fileMenu = new JMenu("File");
        JMenuItem saveMenuItem = new JMenuItem("Save As...");
        saveMenuItem.addActionListener(e -> {
            JFileChooser fileChooser = new JFileChooser();
            if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
                saveImage(fileChooser.getSelectedFile().getAbsolutePath());
            }
        });
        fileMenu.add(saveMenuItem);
        menuBar.add(fileMenu);
        setJMenuBar(menuBar);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(MSPaintButWorse::new);
    }
}
